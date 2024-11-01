import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

data = pd.read_csv('C:/Users/jaolu/Documents/Faculdade/RPAD/RPAD/Lista1/data/artificial1d.csv', names=['x', 'y'])
X = data['x'].values
y = data['y'].values


# OLS
media_x = np.mean(X)
media_y = np.mean(y)

numerador = np.sum((X - media_x) * (y - media_y))
denominador = np.sum((X - media_x) ** 2)
w1_ols = numerador / denominador
w0_ols = media_y - w1_ols * media_x

y_pred_ols = w0_ols + w1_ols * X

mse_ols = np.mean((y - y_pred_ols) ** 2)

# Plotando o grafico
plt.scatter(X, y, color='blue', label='Pontos')
plt.plot(X, y_pred_ols, color='red', label='Linha regressão')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('OLS')
plt.show()

# GD
taxa_aprendizagem = 0.01
n_epocas = 1000
w0_gd, w1_gd = 0.0, 0.0
mse_gd_array = []
frames = []

for epoca in range(n_epocas):
    y_pred_gd = w0_gd + w1_gd * X
    w0_grad = -2 * np.mean(y - y_pred_gd)
    w1_grad = -2 * np.mean((y - y_pred_gd) * X)
    w0_gd -= taxa_aprendizagem * w0_grad
    w1_gd -= taxa_aprendizagem * w1_grad
    mse_gd = np.mean((y - y_pred_gd) ** 2)
    mse_gd_array.append(mse_gd)

    # A cada 50 ele salva o png pra fazer o gif da linha
    if epoca % 50 == 0:
        filename = f"gd_frame_{epoca}.png"
        plt.figure()
        plt.scatter(X, y, color='blue', label='Pontos')
        plt.plot(X, w0_gd + w1_gd * X, color='green', label=f'Epoca {epoca}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(filename)
        plt.close()
        frames.append(filename)

# GIF do GD
with imageio.get_writer('gd.gif', mode='I', duration=0.2) as writer:
    for filename in frames:
        image = imageio.imread(filename)
        writer.append_data(image)

# SGD
w0_sgd, w1_sgd = 0.0, 0.0
mse_sgd_list = []
frames_sgd = []

for epoca in range(n_epocas):
    indices = np.random.permutation(len(X))
    for i in indices:
        xi, yi = X[i], y[i]
        y_pred_sgd = w0_sgd + w1_sgd * xi
        w0_grad = -2 * (yi - y_pred_sgd)
        w1_grad = -2 * (yi - y_pred_sgd) * xi
        w0_sgd -= taxa_aprendizagem * w0_grad
        w1_sgd -= taxa_aprendizagem * w1_grad

    y_pred_sgd_all = w0_sgd + w1_sgd * X
    mse_sgd = np.mean((y - y_pred_sgd_all) ** 2)
    mse_sgd_list.append(mse_sgd)

    # A cada 50 ele salva o png pra fazer o gif da linha
    if epoca % 50 == 0:
        filename = f"sgd_frame_{epoca}.png"
        plt.figure()
        plt.scatter(X, y, color='blue', label='Pontos')
        plt.plot(X, y_pred_sgd_all, color='purple', label=f'Epoca {epoca}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(filename)
        plt.close()
        frames_sgd.append(filename)

# GIF do SGD
with imageio.get_writer('sgd.gif', mode='I', duration=0.2) as writer:
    for filename in frames_sgd:
        image = imageio.imread(filename)
        writer.append_data(image)