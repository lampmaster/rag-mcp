# Kubernetes Deployment

Production workloads run on a managed Kubernetes cluster. This document
describes how to deploy, inspect and roll back a release.

## Namespaces

Every environment has its own namespace. Tooling reads the target namespace
from the K8S_NAMESPACE environment variable, which is set by the environment
profile you source before running any command:

```bash
export K8S_NAMESPACE=platform-staging
```

If K8S_NAMESPACE is unset the scripts refuse to run rather than defaulting to
`default`, because applying a staging manifest to the wrong namespace is an
outage waiting to happen.

## Applying manifests

Manifests live in `deploy/` and are applied with kubectl:

```bash
kubectl apply -f deploy/api-deployment.yaml -n "$K8S_NAMESPACE"
```

`kubectl apply` is declarative: it creates the object if it is missing and
patches it otherwise. Never use `kubectl edit` on a production object - the
change is silently reverted by the next deployment.

## Configuration and secrets

Non sensitive configuration is stored in a ConfigMap and mounted as environment
variables. Sensitive values come from a Secret that is synced from the secret
manager. A ConfigMap change does not restart the pods by itself; roll the
deployment afterwards:

```bash
kubectl rollout restart deployment/api -n "$K8S_NAMESPACE"
```

## Inspecting a release

`kubectl get pods` shows the current state. A pod stuck in `CrashLoopBackOff`
is failing at start up; read the previous container log with
`kubectl logs --previous`. A pod stuck in `Pending` usually means the scheduler
cannot find a node with enough CPU or memory, which is visible in
`kubectl describe pod`.

## Rolling back

Deployments keep the last ten revisions. Undo a bad release with
`kubectl rollout undo deployment/api`. The rollback is a normal rolling update,
so readiness probes still gate the traffic switch.

## Resource requests and limits

Every container declares a CPU and memory request and a memory limit. The
request is what the scheduler reserves; the limit is what the kernel enforces.
A container that exceeds its memory limit is killed with `OOMKilled`, visible
in `kubectl describe pod` under the last state of the container. Requests that
are set far above real usage waste capacity and make the cluster look full
while nodes idle.

## Probes

A readiness probe decides whether a pod receives traffic; a liveness probe
decides whether it is restarted. Point the readiness probe at an endpoint that
also checks the dependencies the pod needs, and point the liveness probe at a
cheap endpoint that only proves the process is alive. Using the same expensive
endpoint for both turns a slow dependency into a restart loop.

## Environment configuration

Application settings are injected as environment variables from a ConfigMap.
The database connection string is one of them: DATABASE_URL is stored in the
ConfigMap for staging and in a Secret for production, since the production URL
embeds a password. Changing a value requires a rollout to take effect, because
environment variables are read once at process start.

Secrets are mounted as files under `/var/run/secrets/platform`, and the
credential used for service to service calls is exposed to the process as an
environment variable by the entrypoint script.

## Autoscaling

The horizontal pod autoscaler adjusts the replica count from the average CPU
utilisation, with a floor of two replicas so that a single node drain never
takes the service down. Scaling up is fast, scaling down is deliberately slow
to avoid flapping. An autoscaler that never scales usually lacks resource
requests - without a request there is no utilisation to compute.

## Ingress and traffic

External traffic enters through the shared ingress controller, which terminates
TLS and routes by hostname. A new hostname needs both an Ingress object and a
DNS record; the certificate is issued automatically once DNS resolves. A `404`
from the ingress with a healthy pod behind it means the host or path rule does
not match.

## Jobs and CronJobs

One-off tasks run as Jobs, scheduled tasks as CronJobs. Set
`activeDeadlineSeconds` on anything that can hang, and keep the history limits
small - completed pods are retained for inspection and otherwise accumulate
until the namespace is full of `Completed` pods.

## Debugging a failing deployment

Start with `kubectl rollout status`, which blocks until the rollout finishes or
fails. Then look at events with `kubectl get events --sort-by=.lastTimestamp`;
image pull failures, missing ConfigMaps and failing probes all surface there
before they show up anywhere else.
