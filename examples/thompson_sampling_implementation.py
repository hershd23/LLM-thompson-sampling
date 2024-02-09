import numpy as np
import io
import cv2
from scipy.stats import beta
import matplotlib.pyplot as plt
import imageio


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



actual_prob = [0.1, 0.7, 0.5]

succ_fail = [[0,0], [0,0], [0,0]]

gif_ims = []

for trials in range(200):

    # plot
    x = np.linspace(0, 1.0, 100)
    y = [beta.pdf(x, s+1, f+1) for s, f in succ_fail]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(x, y[0], "b-", x, y[1], "r-", x, y[2], "g-")
    ax.set(xlabel='payout probabilities', ylabel='PDF', title='Thompson Sampling')
    gif_ims.append(get_img_from_fig(fig))
    plt.close()

    # Sample a data point (thompson sampling) from all arms' Beta distrib
    samples = [np.random.beta(s+1, f+1) for s, f in succ_fail]

    # Pick the arm with highest sampled estimate
    best_arm = np.argmax(samples)

    # Play with best arm
    # since each arm is modelled as bernoulli variable, to sample from bernoulli distribution is same as
    # sampling a uniform distrib variable & comparing with p (payout), if its less than p, then Success else Failure
    if np.random.uniform() < actual_prob[best_arm]:
        # if we win with this arm
        succ_fail[best_arm][0] += 1
    else:
        # if we lose with this arm
        succ_fail[best_arm][1] += 1

    
kwargs_write = {'duration':0.001, 'quantizer':'nq'}
imageio.mimsave('../graphs/final.gif', gif_ims) #, kwargs=kwargs_write)
