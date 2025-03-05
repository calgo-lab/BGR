# SSH Configuration
### Generate SSH Key
```
ssh-keygen
```
Please create a key with passphrase for kubernetes.

### Create SSH Key Configuration in kubernetes secrets from the public key of the system
```
kubectl create secret generic my-ssh-public-key --from-file=authorized_keys=[path_to_public_key(*.pub)]
```

### Create ssh config file
```
nano ~/.ssh/config
```

### Write config in config file
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

# SSH Pod Connection
### Get Pod name
```
kubectl get pods
```

### Port Forward
* Pick a port for your Pod
```
kubectl port-forward [Pod name] 44414:22
```

# Additional Information:

### To clear the pod
```
kubectl scale deployment bgr-project --replicas=0
```

### Screen cheat sheet
https://gist.github.com/jctosta/af918e1618682638aa82