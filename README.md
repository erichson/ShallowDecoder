## Shallow Decoder for flow reconstruction with limited sensors and limited data

Flow reconstruction is an important problem across the physical, biological, and engineering sciences. Given some sensor sensor measurements (orange points)

<img src="https://raw.githubusercontent.com/erichson/ShallowDecoder/master/results/flow_truth_with_sensors.png" width="500">

the aim is it to reconstruct the high-dimensional flow field. Traditionally, techniques such as the proper orthogonal decomposition (POD) are used for this task. Here we propose a neural network-based methodology. Indeed, the reconstruction quality of the shallow decoder outperforms traditional methods. For instance, the following image shows the reconstructed flow field using only 5 sensors: 

<img src="https://raw.githubusercontent.com/erichson/ShallowDecoder/master/results/reconstruction_via_shallow_decoder.png" width="500">


For comparison, the reconstructed flow field using the traditional POD technique:

<img src="https://raw.githubusercontent.com/erichson/ShallowDecoder/master/results/reconstruction_via_pod_plus.png" width="500"> 

### Example

Start by cloning the git repository:

```
git clone https://github.com/erichson/ShallowDecoder
```

Then, you can run a simple examples as follows (without plotting the results):

```
python flow_driver.py --data flow_cylinder --sensor wall --n_sensors 5 --epochs 4000 
```

Or optional you can plot the results:
```
python flow_driver.py --data flow_cylinder --sensor wall --n_sensors 5 --epochs 4000 --plotting True  
```


### Required packages

* [PyTorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org)
* For plotting you need in addition the cmocean package: ```pip install cmocean```


### References

Coming soon!
