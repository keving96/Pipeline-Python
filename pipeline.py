import numpy as np
import matplotlib.pyplot as plt

def twos_complement_test():
    test_list = [x - x*1j for x in range(0,32769)]
    #print(test_list)
    result_list = twos_complement(test_list)
    for i in range(0,32769):
        print(test_list[i].real, test_list[i].imag, result_list[i][0],
            result_list[i][1], int(result_list[i][0], 2),
            int(result_list[i][1], 2), "index:", i)

def twos_complement(inputs):
    real = ""
    imag = ""
    tc_list = []
    for value in inputs:
        # negative number --> (2^16 - number) [-32768, -1]
        # positive number --> (number) [1, 32767]
        
        # real part
        twos_complement_real = round(value.real)
        if twos_complement_real == 0:
            real = format(0, "016b")
        elif twos_complement_real >= -32768 and twos_complement_real < 0:
            real = format(2**16  - abs(twos_complement_real), "016b")
        elif twos_complement_real < 32768 and twos_complement_real > 0:
            real = format(abs(twos_complement_real), "016b")
        # imaginary part
        twos_complement_imag = round(value.imag)
        if twos_complement_imag == 0:
            imag = format(0, "016b")
        elif twos_complement_imag >= -32768 and twos_complement_imag < 0:
            imag = format(2**16  - abs(twos_complement_imag), "016b")
        elif twos_complement_imag < 32768 and twos_complement_imag > 0:
            imag = format(abs(twos_complement_imag), "016b")

        tc_list.append((imag, real))
    return tc_list

def write_to_file(inputs, filename):
    '''
        Writes complex numbers as 32-bit std_logic_vectors into a new file.
        The 32-bit vector is represented as dual 16-bit two's complement numbers
        The first 16 bits form the imaginary part and the last 16 bits the real part.
    '''
    
    targetFile = open(filename + ".txt", "w")

    tc_list = twos_complement(inputs)
    for tc in tc_list:
        targetFile.write(tc[0] + tc[1] + "\n")
    targetFile.close()

def print_list(pList):
    '''Print elements of a list row-wise.'''
    count = 1
    for element in pList:
        print(element, ",", count)
        count += 1

def plot_fft(fft_res):
    '''Plot graph of FFT.'''
    freq = np.fft.fftfreq(fft_res.size)
    #print(freq)
    # plot the results
    plt.plot(freq, fft_res.real, freq, fft_res.imag)
    plt.show()

def pipeline(samples, samples_size):
    '''
        Calculates FFT, its Power Spectrum and IFFT.
    '''
    # input
    print("========================================")
    print("Samples")
    print(samples[0:10])

    # FFT
    print("========================================")
    print("FFT")
    fft_result = np.fft.fft(samples, samples_size, norm = "forward")
    #print_list(fft_result)
    print(fft_result[0:10])
    print("========================================")

    # IFFT
    print("IFFT")
    ifft_result = np.fft.ifft(fft_result, norm = "forward")
    #print_list(ifft_result)
    print(ifft_result[0:10])
    print("========================================")

    # power spectrum
    print("Power Spectrum")
    power_spectrum = (np.abs(fft_result)**2)  
    #print_list(power_spectrum)
    print(power_spectrum[0:10])
    print("========================================")

    # Plot FFT
    #plot_fft(fft_result)

def main():
    '''
        Specify testcases and call methods.
    '''
    #twos_complement_test()

    n = 2**15 # 32768

    # x"7a7a7a7a",
    # x"6b6b6b6b", 
    # x"5c5c5c5c", 
    # x"4d4d4d4d", 
    # x"3e3e3e3e", 
    # x"2b2d2b2d", 
    # x"1a1b1a1b", 
    # x"0b0a0b0a", 
    inputs = [
        (31354 + 31354j),
        (27499 + 27499j),
        (23644 + 23644j),
        (19789 + 19789j),
        (15934 + 15934j),
        (11053 + 11053j),
        (6683 + 6683j),
        (2826 + 2826j)
    ]# * 4096
    #test = [np.random.uniform(7, 2**10) 
    #       + np.random.uniform(7, 2**10) * 1j for _ in range(32678)]
    testcases = np.array(inputs)
    
    # calculate fft, ifft, power spectrum
    print("\nPipeline for self-chosen values")
    #write_to_file(testcases, "testcases")
    #pipeline(testcases, 8)
    
    # sinus random values
    print("\nPipeline for random sinus values")
    sinus = np.sin(np.random.uniform(10*-np.pi, 10*np.pi, n)*1000
                    + np.random.uniform(-np.pi, np.pi, n) * 1j)*1000
    #print(sinus)
    write_to_file(sinus, "sinus")
    #plot_fft(sinus)
    pipeline(sinus, n)
    '''
    # sinus from -pi to +pi
    print("\nPipeline for sinus wave")
    pis = np.linspace(-np.pi, np.pi, n)
    sinus_pis = np.sin(pis + pis * 1j)
    write_to_file(sinus_pis, "sinus_pis")
    print(sinus_pis)
    #plot_fft(sinus_pis)
    #pipeline(sinus_pis, n)
    
    # cosinus random values
    print("\nPipeline for random cosinus values")
    cosinus = np.cos(np.random.uniform(-np.pi, np.pi, n) 
                    + np.random.uniform(-np.pi, np.pi, n) * 1j)
    write_to_file(cosinus, "cosinus")
    print(cosinus)
    #plot_fft(cosinus)
    #pipeline(cosinus, n)

    # cosinus from -pi to +pi
    print("\nPipeline for cosinus wave")
    cosinus_pis = np.cos(pis + pis *1j)
    write_to_file(cosinus_pis, "cosinus_pis")
    print(cosinus_pis)
    #plot_fft(cosinus_pis)
    #pipeline(cosinus_pis, n)'''

if __name__ == "__main__":
    main()