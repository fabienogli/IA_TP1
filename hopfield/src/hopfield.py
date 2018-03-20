from random import randint, shuffle
 
import numpy as np
from matplotlib import pyplot as plt
 
from hopfieldnet.net import  HopfieldNetwork
from hopfieldnet.trainers import hebbian_training
 
#Create the training patterns
a_pattern = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]])
 
b_pattern = np.array([[1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0]])
 
c_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])
 
a_pattern *= 2
a_pattern -= 1
 
b_pattern *= 2
b_pattern -= 1
 
c_pattern *= 2
c_pattern -= 1
 
input_patterns = np.array([a_pattern.flatten(), b_pattern.flatten(), c_pattern.flatten()])
 
#Create the neural network and train it using the training patterns
network = HopfieldNetwork(35)

plt.figure()
plt.suptitle("Exemples d'apprentissage")
plt.subplot(1, 3, 1)
plt.imshow(a_pattern, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(b_pattern, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(c_pattern, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure()
plt.suptitle("Poids avant apprentissage")
plt.imshow(network.get_weights(),interpolation='nearest')
plt.colorbar()
plt.show()
hebbian_training(network, input_patterns)
plt.figure()
plt.suptitle("Poids apres apprentissage")
plt.imshow(network.get_weights(),interpolation='nearest')
plt.colorbar()
plt.show()
 
#Create the test patterns by using the training patterns and adding some noise to them
#and use the neural network to denoise them 


b_test =  b_pattern.flatten()
 
for i in range(20):
    p = randint(0, 34)
    b_test[p] *= -1

plt.figure()
plt.suptitle("Etat initial du reseau (exemple)")
plt.imshow(b_test.reshape((7,5)),interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.show()

b_result = b_test.copy()
changed = True

while changed:
    update_list = range(b_test.size)
    shuffle(update_list)
    changed, b_result = network.run_once(update_list,b_result)
    plt.figure()
    plt.suptitle("Evolution de l'etat du reseau (exemple)")
    plt.imshow(b_result.reshape((7,5)),interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()




a_test =  a_pattern.flatten()
 
for i in range(3):
    p = randint(0, 34)
    a_test[p] *= -1
    
a_result = network.run(a_test,max_iterations=150)
 
a_result.shape = (7, 5)
a_test.shape = (7, 5)
 
b_test =  b_pattern.flatten()
 
for i in range(3):
    p = randint(0, 34)
    b_test[p] *= -1
     
b_result = network.run(b_test)
 
b_result.shape = (7, 5)
b_test.shape = (7, 5)
 
c_test =  c_pattern.flatten()
 
for i in range(3):
    p = randint(0, 34)
    c_test[p] *= -1
     
c_result = network.run(c_test)
 
c_result.shape = (7, 5)
c_test.shape = (7, 5)

d_test =  (np.random.randint(0,2,size=a_test.shape)*2-1).flatten()

print c_test,d_test
   
d_result = network.run(d_test)
 
d_result.shape = (7, 5)
d_test.shape = (7, 5)
 
#Show the results
plt.subplot(4, 2, 1)
plt.title("Entrees bruitees")
plt.imshow(a_test, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.subplot(4, 2, 2)
plt.title("Reseau a convergence")
plt.imshow(a_result, interpolation="nearest")
plt.xticks([])
plt.yticks([])
 
plt.subplot(4, 2, 3)
plt.imshow(b_test, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.subplot(4, 2, 4)
plt.imshow(b_result, interpolation="nearest")
plt.xticks([])
plt.yticks([])
 
plt.subplot(4, 2, 5)
plt.imshow(c_test, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.subplot(4, 2, 6)
plt.imshow(c_result, interpolation="nearest")
plt.xticks([])
plt.yticks([])

plt.subplot(4, 2, 7)
plt.imshow(d_test, interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.subplot(4, 2, 8)
plt.imshow(d_result, interpolation="nearest")
plt.xticks([])
plt.yticks([])
 
plt.show()
