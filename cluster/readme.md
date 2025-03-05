# SSH Key
### Generate SSH Key
```
ssh-keygen
```
Please create a key with passphrase for kubernetes.

### Create SSH Key Configuration in kubernetes secrets from the public key of the system
```
kubectl create secret generic my-ssh-public-key --from-file=authorized_keys=[path_to_public_key(*.pub)]
```

# PVC
### Create PVC:
```
kubectl apply -f storage.yml
```

# Pod
### Creating pods:
```
kubectl apply -f remote-pod.yml
```

# SSH Connection
### Get Pod name
```
kubectl get pods
```

### Port Forward
* Pick a port for your Pod
```
kubectl port-forward [Pod name] 44414:22
```

### Establishing SSH Connection
```
nano ~/.ssh/config
```

### Write confige in config file
* Can create a config for Pod
```
Host kubernetes
    HostName localhost
    Port 44414
    User root
    IdentityFile [path_to_private_key]
```

### To ssh into the pod
```
ssh [ProjectName]
```

# Additional Information:

### To clear the pod
```
kubectl scale deployment bgr-project --replicas=0