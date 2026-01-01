# 🔐 Operational Readiness Guide
**For DevOps / Infrastructure Team**

To move this system from "Demo Mode" to "Production", you must configure the following **Variables** in your GitLab Repository Settings > CI/CD > Variables.

## 1. Container Registry (Docker)
The `.gitlab-ci.yml` builds and pushes the image.
*   `CI_REGISTRY`: (Auto-set by GitLab) `registry.gitlab.com`
*   `CI_REGISTRY_USER`: (Auto-set)
*   `CI_REGISTRY_PASSWORD`: (Auto-set)
*   **Action**: Ensure Container Registry is enabled in your project settings.

## 2. Kubernetes Cluster (Deployment)
The pipeline runs `kubectl apply`. It needs authentication.
*   **Variable Name**: `KUBECONFIG`
*   **Type**: File
*   **Value**: Paste the contents of your `~/.kube/config` file here.
    *   *Security Note*: Ensure this service account has `edit` permissions on the namespace.

## 3. MLflow Tracking (Observability)
To centralize experiment logs from multiple updates:
*   **Variable Name**: `MLFLOW_TRACKING_URI`
*   **Value**: The URL of your remote MLflow server (e.g., `http://mlflow-server.company.com:5000` or Databricks URI).
*   **Variable Name**: `MLFLOW_S3_ENDPOINT_URL` (If using MinIO/S3 for artifacts).

## 4. Production Check
Once these are set, your Pipeline will:
1.  ✅ **Lint & Test** code.
2.  ✅ **Build** Docker Image & Push to Registry.
3.  ✅ **Deploy** to your real Kubernetes Cluster automatically.

NO code changes are required—just this configuration.
