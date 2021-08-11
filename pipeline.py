import numpy as np
import matplotlib.pyplot as plt

def printList(pList):
    count = 1
    for element in pList:
        print(element, ",", count)
        count += 1

def plotFFT(fft_res):
    freq = np.fft.fftfreq(fft_res.size)
    #print(freq)
    # plot the results
    plt.plot(freq, fft_res.real, freq, fft_res.imag)
    plt.show()

def pipeline(samples, samples_size):
    # FFT
    print("========================================")
    print("FFT")
    fft_result = np.fft.fft(samples, samples_size, norm = "backward")
    #printList(fft_result)
    print(fft_result[0:30])
    print("========================================")

    # IFFT
    print("IFFT")
    ifft_result = np.fft.ifft(fft_result)
    #printList(ifft_result)
    print(ifft_result)
    print("========================================")

    # power spectrum
    print("Power Spectrum")
    power_spectrum = (np.abs(fft_result)**2)  
    #printList(power_spectrum)
    print(power_spectrum)
    print("========================================")

    # Plot FFT
    #plotFFT(fft_result)

def main():
    n = 2**15 # 32768
    start = 1
    step = 2**10
    #complex_array = values * np.arange(start, n, step) # numpy.ndarray

    # x"aaaaaaaa", x"bbbbbbbb", x"cccccccc", x"dddddddd", x"eeeeeeee", x"abcdabcd", x"aabbaabb", x"bbaabbaa", 
    inputs = [
        1+0j,#(43690 + 43690j),
        1+0j,#(48059 + 48059j),
        1+0j,#(52428 + 52428j),
        1+0j,#(56797 + 56797j),
        1+0j,#(61166 + 61166j),
        1+0j,#(43981 + 43981j),
        1+0j,#(43707 + 43707j),
        1+0j,#(48042 + 48042j)
    ]
    testcases = np.array(inputs)
    #print(complex_array)
    # complex_array.size === stop / step

    # calculate fft, ifft, power spectrum
    print("\nPipeline for self-chosen values")
    pipeline(testcases, n)
    '''
    # sinus random values
    print("\nPipeline for random sinus values")
    sinus = np.sin( np.random.uniform(-np.pi, np.pi, n) + np.random.uniform(-np.pi, np.pi, n) * 1j )
    print(sinus)
    #plotFFT(sinus)
    pipeline(sinus, n)

    # sinus from -pi to +pi
    print("\nPipeline for sinus wave")
    pis = np.linspace(-np.pi, np.pi, n)
    sinus_pis = np.sin(pis + pis * 1j)
    print(sinus_pis)
    #plotFFT(sinus_pis)
    pipeline(sinus_pis, n)

    # cosinus random values
    print("\nPipeline for random cosinus values")
    cosinus = np.cos( np.random.uniform(-np.pi, np.pi, n) + np.random.uniform(-np.pi, np.pi, n) * 1j )
    print(cosinus)
    #plotFFT(cosinus)
    pipeline(cosinus, n)

    # cosinus from -pi to +pi
    print("\nPipeline for cosinus wave")
    cosinus_pis = np.cos(pis + pis *1j)
    print(cosinus_pis)
    #plotFFT(cosinus_pis)
    pipeline(cosinus_pis, n)'''

if __name__ == "__main__":
    main()