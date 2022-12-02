
import matplotlib.pyplot as plt
import numpy as np

import copy,math

def plot_loss_tf(history):
    losses = history.history['loss']
    plt.plot(losses)
    try:
        val_loss = history.history['val_loss']
        plt.plot(val_loss)
        plt.legend(['loss','test_loss'])
    except:
        pass

    plt.xlabel('epoch')
    plt.ylabel('losses')

def normalize_data(data):
    mean = data.mean()
    max_minus_min = data.max()-data.min()
    data = (data-mean)/max_minus_min
    return data, mean, max_minus_min

def unnormalized_data(norm_price,mean_price,max_minus_min_price):

    return norm_price*max_minus_min_price+mean_price

def calculate_errors(model,X,y,E,price_error,mean_price,max_minus_min_price):
    failed_X = []
    failed_y = []

    test_size = X.shape[0]
    prediction = np.exp(unnormalized_data(model.predict(X),mean_price,max_minus_min_price))
    y = np.exp(unnormalized_data(y,mean_price,max_minus_min_price))



    # for i in range(len(prediction)):
    #     print(prediction[i],y[i])
    #     print('error = ',abs(prediction[i]-y[i])/prediction[i]*100,'%')
    failed_X = [X[i,:] for i in range(len(prediction)) if not prediction[i]-prediction[i]*price_error<=y[i]<=prediction[i]+prediction[i]*price_error]
    failed_prediction = [prediction[i] for i in range(len(prediction)) if not prediction[i]-prediction[i]*price_error<=y[i]<=prediction[i]+prediction[i]*price_error]
    failed_y = [y[i] for i in range(len(prediction)) if not prediction[i]-prediction[i]*price_error<=y[i]<=prediction[i]+prediction[i]*price_error]
    percent_failed = round(len(failed_X)/test_size*100*100)/100
    print(f'{percent_failed} % of failure (or {len(failed_X)} out of {test_size})')

    #append to E_cv or E_train
    E.append(percent_failed)

    return np.array(failed_X),np.array([failed_prediction]).T,np.array([failed_y]).T,E


def carth_to_polar(x,y):

    x0 = 43.647144
    y0 = -79.381204 #Toronto Union station

    r = np.sqrt((x-x0)**2+(y-y0)**2)
    theta = np.arctan((x-x0)/(y-y0))
    return r,theta


# Regression function
def prediction(X,w,b):
    return np.dot(X,w)+b

def compute_cost(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar
    return cost

def compute_gradient(X, y, w, b):
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):

        err = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]}")

    return w, b, J_history #return final w,b and J history for graphing
