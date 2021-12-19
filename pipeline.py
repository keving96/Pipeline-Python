import numpy as np
import matplotlib.pyplot as plt
import itertools as itools
import bitstring
import math

def single_precision_floating_point(numbers):
    fp_list = []
    for number in numbers:
        real = bitstring.BitArray(float=number.real, length=32)
        imag = bitstring.BitArray(float=number.imag, length=32)
        fp_list.append((imag.bin, real.bin))
    return fp_list

def write_to_file_floating_point(inputs, filename):
    '''
        Writes a 64-bit floating point representation into a new file.
        The first 32 bits form the imaginary part and the last 32 bits the real part.
    '''
    targetFile = open(filename + ".txt", "w")

    fp_list = single_precision_floating_point(inputs)
    for fp in fp_list:
        targetFile.write(fp[0] + fp[1] + "\n")
    targetFile.close()

def twos_complement_test():
    test_list = [x - x*1j for x in range(4294960295,4294967297)]
    result_list = twos_complement(test_list)
    for i in range(0,7002):
        print(test_list[i].real, test_list[i].imag, result_list[i][0],
            result_list[i][1], int(result_list[i][0], 2),
            int(result_list[i][1], 2), "index:", i)

def twos_complement(inputs):
    real = ""
    imag = ""
    tc_list = []
    for value in inputs:
        # negative number --> (2^32 - number) [-4294967296, -1]
        # positive number --> (number) [1, 4294967295]

        # real part
        twos_complement_real = round(value.real)
        if twos_complement_real == 0:
            real = format(0, "032b")
        elif twos_complement_real >= -(2**32) and twos_complement_real < 0:
            real = format(2**32  - abs(twos_complement_real), "032b")
        elif twos_complement_real < (2**32) and twos_complement_real > 0:
            real = format(abs(twos_complement_real), "032b")

        # imaginary part
        twos_complement_imag = round(value.imag)
        if twos_complement_imag == 0:
            imag = format(0, "032b")
        elif twos_complement_imag >= -(2**32) and twos_complement_imag < 0:
            imag = format(2**32  - abs(twos_complement_imag), "032b")
        elif twos_complement_imag < (2**32) and twos_complement_imag > 0:
            imag = format(abs(twos_complement_imag), "032b")

        tc_list.append((imag, real))
    return tc_list

def write_to_file_fixed_point(inputs, filename):
    '''
        Writes complex numbers as 64-bit std_logic_vectors into a new file.
        The 64-bit vector is represented as dual 32-bit two's complement numbers
        The first 32 bits form the imaginary part and the last 32 bits the real part.
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
        Calculates FFT, its Power Spectrum, IFFT and accumulates
        Power Spectrum
    '''
    # input
    print("========================================")
    print("Samples")
    print(samples[0:20])
    plot_fft(samples)
    # FFT
    print("========================================")
    print("FFT")
    fft_result = np.fft.fft(samples, samples_size, norm = "backward")
    #print_list(fft_result)
    print(fft_result[0:20])
    print("========================================")
    plot_fft(fft_result)
    # IFFT
    print("IFFT")
    ifft_result = np.fft.ifft(fft_result, norm = "backward")
    #print_list(ifft_result)
    print(ifft_result[0:20])
    print("========================================")

    # power spectrum
    print("Power Spectrum")
    power_spectrum = (np.abs(fft_result)**2)  
    #print_list(power_spectrum)
    print(power_spectrum[0:20])
    print(max(power_spectrum))
    print("========================================")

    print("acccumulation with itertools")
    #for each in itools.accumulate(power_spectrum[0:samples_size]):
    #    print(each)
    akkum = list(itools.accumulate(power_spectrum[0:samples_size]))
    print(akkum[0:60])
    print("########################################")
    
    # Plot FFT
    #plot_fft(fft_result)

def main():
    '''
        Specify testcases and call methods.
    '''
    #twos_complement_test()

    n = 2**15 # 32768

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

    testcases = np.array(inputs)
    
    # calculate fft, ifft, power spectrum
    print("\nPipeline for self-chosen values")
    #write_to_file_fixed_point(testcases, "testcases")
    #write_to_file_floating_point(testcases, "testcases")
    #pipeline(testcases, 8)

    # sinus random values
    '''print("\nPipeline for random sinus values")
    sinus = np.sin(np.random.uniform(10*-np.pi, 10*np.pi, n)*1000
                    + np.random.uniform(-np.pi, np.pi, n) * 1j)*1000
    print(sinus)
    write_to_file_floating_point(sinus, "sinus")
    #plot_fft(sinus)
    pipeline(sinus, n)'''

    # sinus from -pi to +pi
    print("\nPipeline for sinus wave")
    pis = np.linspace(0, 20*np.pi, n)
    sinus_pis = np.sin(pis)*1234 + np.sin(pis)*4321 * 1j
    print(sinus_pis)
    #write_to_file_floating_point(sinus_pis, "sinus_pis")

    #plot_fft(sinus_pis)
    pipeline(sinus_pis, n)
    '''
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