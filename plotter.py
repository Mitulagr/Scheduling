import matplotlib.pyplot as plt
#from IPython import display

#plt.ion()

def plot(scores,fifo,sjf) : 

    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    # plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Score')
    plt.plot(scores)
    #plt.plot(mean_scores)
    plt.plot(fifo)
    plt.plot(sjf)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1,scores[-1],str(scores[-1]))
    plt.text(len(fifo)-1,fifo[-1],str(fifo[-1]))
    plt.text(len(sjf)-1,sjf[-1],str(sjf[-1]))
    plt.show()