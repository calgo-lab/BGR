# SSH Key
### Create SSH Key Configuration in kubernetes secrets from the public key of the system
```
kubectl create secret generic my-ssh-public-key --from-file=authorized_keys=[path to public key]
```

# PVC
### Create PVC:
```
kubectl apply -f storage.yml
```

# Pod
### Creating pods:
```
kubectl apply -n my-namespace -f remote-pod.yaml
```

# SSH Connection
### Get Pod name
```
kubectl get pods
```

### Port Forward
* Pick a port for your Pod
```
kubectl port-forward [PodName] 44444:22
```

### Establishing SSH Connection
```
nano ~/.ssh/config
```

### Write confige in config file
* Can create a config for Pod
```
Host [ProjectName]
    HostName localhost
    Port [Port]
    User root
    IdentityFile ~/.ssh/id_rsa
```

### To ssh into the pod
```
ssh [ProjectName]
```

# Additional Information:

### To clear the pod
```
kubectl scale deployment bgr-project --replicas=0