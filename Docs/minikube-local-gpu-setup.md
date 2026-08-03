# Local Minikube + GPU setup log

Goal: validate the Kubernetes deployment plumbing for the NeMo tool-calling
project on a local, free, single-node cluster (Minikube) before spending money
on an NVIDIA Brev Launchable. This doc is a running log of exactly what was
run and why, in order, so the steps can be reproduced or audited later.

## 0. System state before starting

Checked on 2026-08-03.

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Arch | x86_64 |
| Docker | 29.6.2 (rootless-capable, user already in `docker` group) |
| Docker default runtime | `runc`, with an `nvidia` runtime already registered in `/etc/docker/daemon.json` |
| GPU | NVIDIA GeForce GTX 1650 Ti, 4096 MiB VRAM |
| NVIDIA driver | 580.173.02 |
| nvidia-container-toolkit | 1.18.2-1 (already installed) |
| kubectl | not installed |
| minikube | not installed |
| sudo | requires a password (not passwordless) |

Because `docker ps` already works without `sudo` for this user, and
`nvidia-container-toolkit` + the Docker `nvidia` runtime are already present,
the plan is to install `kubectl` and `minikube` as plain user binaries in
`~/.local/bin` (already on `$PATH`) so **no `sudo` / password prompts are
needed at all**.

Command used to gather the above:

```bash
cat /etc/os-release
docker --version
docker info | grep -i "server version\|cgroup\|runtime"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
dpkg -l | grep -i nvidia-container
cat /etc/docker/daemon.json
which kubectl
sudo -n true   # tests for passwordless sudo -> failed, so we avoid sudo entirely
docker ps       # confirms docker works without sudo
groups          # confirms user is in the `docker` group
```

## 1. Install kubectl

Installed as a plain user binary (no package manager, no sudo), matching the
current stable Kubernetes release.

```bash
KVER=$(curl -Ls https://dl.k8s.io/release/stable.txt)   # -> v1.36.3
curl -LO "https://dl.k8s.io/release/${KVER}/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/release/${KVER}/bin/linux/amd64/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check   # -> "kubectl: OK"
install -m 0755 kubectl ~/.local/bin/kubectl
```

Verified:

```bash
$ kubectl version --client
Client Version: v1.36.3
Kustomize Version: v5.8.1
```

## 2. Install minikube

Same approach as kubectl: plain binary into `~/.local/bin`, checksum-verified,
no sudo.

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64.sha256
echo "$(cat minikube-linux-amd64.sha256)  minikube-linux-amd64" | sha256sum --check   # -> OK
install -m 0755 minikube-linux-amd64 ~/.local/bin/minikube
```

Verified:

```bash
$ minikube version
minikube version: v1.38.1
commit: c93a4cb9311efc66b90d33ea03f75f2c4120e9b0
```

## 3. Start minikube with GPU passthrough

First attempt:

```bash
minikube start --driver=docker --container-runtime=docker --gpus=all
```

`--gpus` requires the Docker driver *and* Docker as the in-cluster container
runtime (not containerd), which is why both flags are set.

**First result: failed.** `minikube status` reported:

```
host: InsufficientStorage
kubelet: Stopped
apiserver: Stopped
```

Root cause (not a minikube problem — a pre-existing disk problem):

```bash
$ df -h /
Filesystem                 Size  Used Avail Use% Mounted on
/dev/mapper/vgubuntu-root  467G  438G  5.1G  99% /

$ docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          65        15        40.44GB   17.08GB (42%)
Containers      40        1         2.23GB    2.23GB (99%)
Local Volumes   50        14        8.015GB   6.467GB (80%)
Build Cache     561       0         61.24GB   42.31GB
```

Root filesystem is at 99% (5.1GB free). Docker's build cache alone (42GB
reclaimable) and old stopped containers from unrelated projects on this
machine (`wms-*`, `shahname-api-*`, various `ros`/`ov_ros2` containers,
`remote_vlm_deployment-*`) are eating the space. minikube's pre-flight check
refuses to provision a node without enough headroom.

Cleaned up the failed partial cluster before doing anything else (this only
removes minikube's own container, nothing belonging to other projects):

```bash
$ minikube delete
* Deleting "minikube" in docker ...
* Deleting container "minikube" ...
* Removing /home/roboticistprogrammer/.minikube/machines/minikube ...
* Removed all traces of the "minikube" cluster.
```

**Paused here** to ask the user how they want to reclaim space, since the
reclaimable Docker data belongs to several *other* unrelated projects on this
machine and shouldn't be deleted without confirmation.

User chose to clear only the Docker build cache (safest option — doesn't
touch stopped containers or images from other projects on this machine):

```bash
$ docker builder prune -a -f
Total:  61.24GB

$ df -h /
Filesystem                 Size  Used Avail Use% Mounted on
/dev/mapper/vgubuntu-root  467G  341G  103G  77% /
```

Freed 61GB, now 103GB free. Retrying:

```bash
$ minikube start --driver=docker --container-runtime=docker --gpus=all
* minikube v1.38.1 on Ubuntu 22.04
* Using the docker driver based on user configuration
* Using Docker driver with root privileges
* Starting "minikube" primary control-plane node in "minikube" cluster
* Pulling base image v0.0.50 ...
* Configuring bridge CNI (Container Networking Interface) ...
* Verifying Kubernetes components...
  - Using image gcr.io/k8s-minikube/storage-provisioner:v5
  - Using image nvcr.io/nvidia/k8s-device-plugin:v0.18.2
* Enabled addons: nvidia-device-plugin, storage-provisioner, default-storageclass
* Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default
```

Note `--gpus=all` automatically enabled the `nvidia-device-plugin` addon — this
is the daemonset that advertises the host GPU as an allocatable Kubernetes
resource (`nvidia.com/gpu`) and is what lets the scheduler place GPU-requesting
pods correctly.

Verified the cluster and the GPU resource:

```bash
$ minikube status
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured

$ kubectl get nodes -o wide
NAME       STATUS   ROLES           AGE   VERSION   CONTAINER-RUNTIME
minikube   Ready    control-plane   56s   v1.35.1   docker://29.2.1

$ kubectl get pods -n kube-system | grep nvidia
nvidia-device-plugin-daemonset-bg522   1/1   Running   0   47s

$ kubectl get node minikube -o jsonpath='{.status.capacity}'
{
  "cpu": "8",
  "ephemeral-storage": "488644656Ki",
  "memory": "7919512Ki",
  "nvidia.com/gpu": "1",
  "pods": "110"
}
```

`"nvidia.com/gpu": "1"` on the node confirms the cluster sees the host's GTX
1650 Ti as a schedulable resource. That's proof the *plumbing* is wired
correctly — next step is proving a pod can actually use it, not just that the
node advertises it.

## 4. Verify GPU passthrough with a test pod

Advertising `nvidia.com/gpu` on the node isn't proof a container can actually
*use* the GPU — that requires the device files and CUDA driver libraries to
be correctly mounted into the container by the runtime. So we ran a real pod
that requests the GPU and calls `nvidia-smi` from inside the container.

Manifest (`gpu-test-pod.yaml`, a throwaway smoke test, not committed to the
repo):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
    - name: cuda-smi
      image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

```bash
kubectl apply -f gpu-test-pod.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/gpu-test --timeout=180s
kubectl logs gpu-test
```

Result — the pod completed successfully and `nvidia-smi`, run **from inside
a container scheduled by Kubernetes**, correctly reported the physical host
GPU:

```
NAME       READY   STATUS      RESTARTS   AGE
gpu-test   0/1     Completed   0          21s

Mon Aug  3 12:08:05 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce GTX 1650 Ti     Off |   00000000:01:00.0  On |                  N/A |
| N/A   54C    P8              6W /   50W |      63MiB /   4096MiB |     26%      Default |
+-----------------------------------------+------------------------+----------------------+
```

This is the actual end-to-end proof: Kubernetes scheduler → device plugin →
container runtime → CUDA driver → physical GPU, all working together on a
completely free, local cluster.

Cleaned up:

```bash
kubectl delete pod gpu-test
```

## 5. Takeaways

- **The orchestration layer is fully verified end-to-end, for free.** A
  Deployment/Pod manifest that requests `nvidia.com/gpu` will schedule and run
  correctly on this local cluster exactly the same way it would on a real
  Brev Launchable node — same Kubernetes API, same scheduler logic, same
  device-plugin mechanism. This is the "plumbing" layer: container image
  builds, manifests, GPU resource requests/limits, networking, all testable
  here with unlimited free iterations.

- **The physical hardware ceiling does not move.** This machine's GTX 1650 Ti
  only has 4096MiB (4GB) VRAM, confirmed directly in the `nvidia-smi` output
  above. No amount of Kubernetes/Minikube configuration changes that — Layer
  2 (actual GPU compute/memory) is hardware pass-through, not virtualized.
  Fine-tuning/inference jobs that need more VRAM than 4GB will still OOM
  locally regardless of orchestration correctness.

- **Recommended workflow going forward:** use this Minikube cluster to
  iterate for free on deployment configs, container images, and small/
  quantized smoke tests of the NeMo tool-calling pipeline. Only spin up the
  paid Brev Launchable for the runs that actually need its larger GPU's
  memory/compute — so money is spent on real GPU-bound work, not on
  debugging YAML.

- **Useful commands for next time:**
  ```bash
  minikube start --driver=docker --container-runtime=docker --gpus=all   # start
  minikube stop                                                          # pause (keeps state, frees CPU/RAM)
  minikube delete                                                        # tear down completely
  kubectl get node minikube -o jsonpath='{.status.capacity}'             # confirm GPU still visible
  ```
