# Locally Weighted Regression
* Non-parametric form of regression
* Instead of fitting a line to the whole data set, each input is considered individually and a best fit of the parameters is calculated for that specific input 
* A different set of parameter is assigned to every point
* The values of the parameters are assigned based on the weights of the neighboring inputs (therefore, the name "Locally weighted")

The way the LWR works in this code is:
1) Take a point in x
2) Calculate the weights of every other point w.r.t to the point taken into consideration
3) Save these weights in the Diagonal Weighted Matrix
4) Reduce the cost function and find theta that best fits the data for the chosen point and its neighboring area
5) Save all the thetas for all the different points
6)Predict the output