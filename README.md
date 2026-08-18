Step 1 — Prerequisites + Plan

!/bin/bash
scripts/personalization-precheck.sh
set -euo pipefail

PASS=0; FAIL=0

check() {
  local desc=$1 cmd=$2
  if eval "$cmd" &>/dev/null; then
    echo "✅ $desc"; ((PASS++))
  else
    echo "❌ $desc"; ((FAIL++))
  fi
}

check "kubectl connected"       "kubectl get nodes"
check "production namespace"    "kubectl get namespace production"
check "redis running"           "kubectl get pods -n production | grep cache | grep Running"
check "ml-serve running"        "kubectl get pods -n production | grep ml-serve | grep Running"
check "prometheus healthy"      "curl -sf http://prometheus:9090/-/healthy"
check "prod health"             "curl -sf https://prod.placemux.com/health"
check "opensearch healthy"      "curl -sf https://<opensearch-endpoint>/_cluster/health"
check "mlflow accessible"       "curl -sf http://mlflow:5000/health"
check "aws cost explorer"       "aws ce get-cost-and-usage --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) --granularity MONTHLY --metrics BlendedCost"

echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ $FAIL -eq 0 ] || exit 1

 docs/personalization-plan.md

 Stage A — The bar
Personalized responses stay fast and affordable at peak load.

 SLO targets
- p50 latency: <50ms
- p95 latency: <200ms
- p99 latency: <500ms
- Cost per request: <$0.001
- Cache hit rate: >80%

 Blast radius

| Stage | What changes               | Blast radius                      | Rollback                          |
|-------|----------------------------|-----------------------------------|-----------------------------------|
| B     | Feature cache + Redis      | Cold cache on restart             | Flush + repopulate from DB        |
| C     | Inference autoscaling      | Over/under-scale during tuning    | kubectl apply previous hpa yaml   |
| D     | Cost metrics               | Metrics only — no infra change    | Remove metric instrumentation     |
| E     | Live load test             | Real inference traffic on prod    | Kill switch on personalization    |

bash scripts/personalization-precheck.sh

Step 2 — Feature cache (Stage B)

-- migrations/020_personalization.sql
CREATE TABLE user_features (
  user_id         TEXT PRIMARY KEY,
  tenant_id       UUID NOT NULL,
  preferred_locs  TEXT[],
  salary_min      INTEGER,
  salary_max      INTEGER,
  top_skills      TEXT[],
  click_history   JSONB,
  last_active     TIMESTAMPTZ,
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE personalization_log (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         TEXT NOT NULL,
  request_type    TEXT NOT NULL,
  cache_hit       BOOLEAN NOT NULL,
  latency_ms      INTEGER NOT NULL,
  model_version   TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_features_tenant ON user_features(tenant_id);
CREATE INDEX idx_personalization_log  ON personalization_log(user_id, created_at);

 src/personalization/features.js
const redis  = require('redis');
const { Pool } = require('pg');

const cache = redis.createClient({ url: redis://${process.env.REDIS_HOST}:6379 });
const db    = new Pool({ connectionString: process.env.DATABASE_URL });

const TTL = 300;

cache.connect();

const getFeatures = async (userId) => {
  const key    = features:${userId};
  const cached = await cache.get(key);
  if (cached) return { features: JSON.parse(cached), cacheHit: true };

  const { rows: [user] } = await db.query(
    'SELECT * FROM user_features WHERE user_id=$1', [userId]
  );

  if (!user) {
    const defaults = { preferred_locs: [], salary_min: 0,
                       salary_max: 999999, top_skills: [] };
    await cache.setEx(key, TTL, JSON.stringify(defaults));
    return { features: defaults, cacheHit: false };
  }

  await cache.setEx(key, TTL, JSON.stringify(user));
  return { features: user, cacheHit: false };
};

const updateFeatures = async (userId, tenantId, event) => {
  await db.query(
    INSERT INTO user_features (user_id, tenant_id, click_history, last_active, updated_at)
     VALUES ($1,$2,$3::jsonb,NOW(),NOW())
     ON CONFLICT (user_id) DO UPDATE
     SET click_history = user_features.click_history || $3::jsonb,
         last_active   = NOW(),
         updated_at    = NOW(),
    [userId, tenantId, JSON.stringify({ [event.job_id]: event.ts })]
  );
  await cache.del(features:${userId});
};

const warmCache = async (tenantId) => {
  const { rows } = await db.query(
    'SELECT * FROM user_features WHERE tenant_id=$1 AND last_active > NOW()-INTERVAL $2',
    [tenantId, '24 hours']
  );
  for (const user of rows) {
    await cache.setEx(features:${user.user_id}, TTL, JSON.stringify(user));
  }
  console.log(JSON.stringify({ warmed: rows.length, tenant: tenantId }));
};

module.exports = { getFeatures, updateFeatures, warmCache };

 k8s/production/cache-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: production
data:
  redis.conf: |
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    save ""
    appendonly no
    tcp-keepalive 60
    timeout 300
    
kubectl apply -f k8s/production/cache-config.yaml
kubectl apply -f k8s/production/cache-warmer-cronjob.yaml

Step 3 — Inference autoscaling (Stage C)

ml/serve.py — extended with batch + timeout
import os, json, time, joblib, mlflow.sklearn
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import pandas as pd

app = Flask(__name__)

PRED_COUNT   = Counter('ml_predictions_total',    'Predictions',   ['model','version','cache_hit'])
PRED_LATENCY = Histogram('ml_prediction_seconds', 'Latency',       ['model'],
               buckets=[.005,.01,.025,.05,.1,.2,.5,1,2])
PRED_ERRORS  = Counter('ml_prediction_errors_total','Errors',      ['model'])
QUEUE_SIZE   = Gauge('ml_request_queue_size',     'Queue depth')

MODEL_NAME = os.environ.get('MODEL_NAME', 'job-ranker')
model      = None
version    = 'unknown'

def load_model():
    global model, version
    client  = mlflow.tracking.MlflowClient(os.environ['MLFLOW_TRACKING_URI'])
    prod    = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
               if v.current_stage == 'Production']
    version = prod[0].version if prod else 'unknown'
    model   = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': MODEL_NAME, 'version': version})

@app.route('/predict', methods=['POST'])
def predict():
    QUEUE_SIZE.inc()
    start = time.time()
    try:
        data = request.json
        jobs = data.get('jobs', [])
        if not jobs:
            return jsonify({'scores': [], 'version': version})
        df    = pd.DataFrame(jobs)
        X     = pd.get_dummies(df[['salary','location']].fillna(0))
        preds = model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') \
                else model.predict(X)
        scores = [{'id': j['id'], 'score': float(s)}
                  for j, s in zip(jobs, preds)]
        latency = time.time() - start
        PRED_LATENCY.labels(MODEL_NAME).observe(latency)
        PRED_COUNT.labels(MODEL_NAME, version,
                          str(data.get('cache_hit', False))).inc()
        return jsonify({'scores': scores, 'version': version, 'latency': latency})
    except Exception as err:
        PRED_ERRORS.labels(MODEL_NAME).inc()
        return jsonify({'error': str(err)}), 500
    finally:
        QUEUE_SIZE.dec()

@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/reload', methods=['POST'])
def reload():
    load_model()
    return jsonify({'reloaded': True, 'version': version})

load_model()
app.run(host='0.0.0.0', port=8080, threaded=True)

kubectl apply -f k8s/production/ml-serve-hpa.yaml

Step 4 — Cost + latency measurement (Stage D)

src/personalization/metrics.js
const client = require('prom-client');

const personalizationLatency = new client.Histogram({
  name:       'personalization_latency_seconds',
  labelNames: ['cache_hit', 'tenant'],
  buckets:    [.005,.01,.025,.05,.1,.2,.5,1]
});

const personalizationCost = new client.Counter({
  name:       'personalization_cost_usd_total',
  labelNames: ['tenant']
});

const cacheHitRate = new client.Gauge({
  name:       'personalization_cache_hit_rate',
  labelNames: ['tenant']
});

const COST_PER_ML_CALL    = 0.0001;
const COST_PER_CACHE_HIT  = 0.000001;

const recordRequest = async (tenantId, cacheHit, latencyMs) => {
  personalizationLatency
    .labels(String(cacheHit), tenantId)
    .observe(latencyMs / 1000);

  const cost = cacheHit ? COST_PER_CACHE_HIT : COST_PER_ML_CALL;
  personalizationCost.labels(tenantId).inc(cost);
};

module.exports = { personalizationLatency, personalizationCost,
                   cacheHitRate, recordRequest };

!/bin/bash
scripts/cost-per-request.sh
set -euo pipefail

echo "=== ML serving cost ==="
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=TAG,Key=component \
  | jq '.ResultsByTime[0].Groups[] | select(.Keys[0]=="component$ml-serve")'

echo "=== requests today ==="
curl -s 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=sum(increase(ml_predictions_total[24h]))' \
  | jq -r '.data.result[0].value[1]'

echo "=== cost per request ==="
curl -s 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=sum(increase(personalization_cost_usd_total[24h]))
                         / sum(increase(ml_predictions_total[24h]))' \
  | jq -r '"$" + .data.result[0].value[1]'

echo "=== cache hit rate ==="
curl -s 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=sum(increase(ml_predictions_total{cache_hit="true"}[1h]))
                         / sum(increase(ml_predictions_total[1h])) * 100' \
  | jq -r '.data.result[0].value[1] + "%"'

echo "=== p95 latency ==="
curl -s 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=histogram_quantile(0.95,
    rate(personalization_latency_seconds_bucket[1h])) * 1000' \
  | jq -r '.data.result[0].value[1] + "ms"'

  kubectl apply -f k8s/production/personalization-alerts.yaml

Step 5 — End-to-end demo (Stage E)

 tests/load/personalization.js
import http  from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100  },
    { duration: '5m', target: 500  },
    { duration: '3m', target: 1000 },
    { duration: '2m', target: 0    },
  ],
  thresholds: {
    http_req_duration:                              ['p(50)<50','p(95)<200','p(99)<500'],
    http_req_failed:                                ['rate<0.001'],
    'http_req_duration{scenario:cached}':           ['p(95)<50'],
    'http_req_duration{scenario:uncached}':         ['p(95)<200'],
  },
};

const TENANTS = ['acme','globex','initech'];

export default function () {
  const tenant = TENANTS[Math.floor(Math.random() * TENANTS.length)];
  const params = {
    headers: { Authorization: Bearer ${__ENV.TOKEN} },
    tags:    { scenario: Math.random() < 0.8 ? 'cached' : 'uncached' }
  };

  const res = http.get(
    https://${tenant}.placemux.com/jobs/search?q=engineer&personalized=true,
    params
  );
  check(res, {
    'status 200':          r => r.status === 200,
    'has variant':         r => JSON.parse(r.body).jobs?.length > 0,
    'latency <200ms':      r => r.timings.duration < 200,
  });
  sleep(1);
}

bash scripts/personalization-demo.sh

Expected output:

pre-check          all passed ✅
cache warmed       N keys in Redis
1st request        cache_hit=false latency_ms ~150
2nd request        cache_hit=true  latency_ms ~8
load test          p50<50ms p95<200ms p99<500ms
SLO                p95 <200ms confirmed under 1000 VUs
cost/request       <$0.001
cache hit rate     >80%
cold cache         p95 spikes to ~400ms, CacheHitRateLow alert fires
rewarm             p95 returns to <200ms
autoscaling        ml-serve 2→N pods during load, scales back after

<img width="1536" height="1024" alt="ss3012" src="https://github.com/user-attachments/assets/105e61c6-775f-48d6-887b-c1e4a3663d49" />
