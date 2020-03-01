#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "iostream"
#include <ctime>

#define liczba_blokow 16
#define liczba_watkow 512
#define iloczyn 8192

cudaError_t addWithCuda(unsigned long long liczba, bool czy_niepierwsza_cuda, unsigned long long pierwiastek);

__global__ void is_prime_Kernel(unsigned long long liczba, bool &czy_niepierwsza, unsigned long long pierwiastek)
{
	unsigned long long idx = threadIdx.x + 2 + blockIdx.x * blockDim.x;
	while (idx <= pierwiastek) {
		if (liczba % idx == 0) czy_niepierwsza = true;
		idx = idx + iloczyn;
	}
}

int main()
{
	bool czy_niepierwsza = false;
	unsigned long long liczba;
	unsigned long long pierwiastek;

	bool czy_niepierwsza_cuda = false;

	printf("Podaj liczbe: \n");
	//scanf("%llu", &potential_prime);
	liczba = 100000150499;
	printf("Twoja liczba to: %llu\n", liczba);
	pierwiastek = sqrt(liczba);

	std::cout << "(CUDA): ";

	clock_t start = clock();
	if (pierwiastek * pierwiastek == liczba) {
		std::cout << "Liczba " << liczba << " nie jest liczba pierwsza poniewaz dzieli sie przez pierwiastek " << pierwiastek << "\n";
		printf("Czas wykonywania: %lu ms\n", clock() - start);
		czy_niepierwsza = true;
	}
	int dzielnik = 1;
	if (!czy_niepierwsza) {
		do {
			dzielnik++;
		} while (liczba % dzielnik != 0 && dzielnik < pierwiastek + 1);
		if (dzielnik == pierwiastek + 1) {
			std::cout << "Liczba " << liczba << " jest liczba pierwsza \n";
			printf("Czas wykonywania: %lu ms\n", clock() - start);
		}
		else {
			std::cout << "Liczba " << liczba << " nie jest liczba pierwsza, poniewaz dzieli sie przez " << dzielnik << "\n";
			printf("Czas wykonywania: %lu ms\n", clock() - start);
		}
	}

	std::cout << "(CUDA): ";

	cudaError_t cudaStatus = addWithCuda(liczba, czy_niepierwsza_cuda, pierwiastek);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	return 0;
}

cudaError_t addWithCuda(unsigned long long liczba, bool czy_niepierwsza_cuda, unsigned long long pierwiastek)
{
	bool* czy_niepier;
	float computing_time = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start_counting, stop_counting;

	cudaStatus = cudaEventCreate(&start_counting);
	cudaStatus = cudaEventCreate(&stop_counting);

	cudaEventRecord(start_counting, 0);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&czy_niepier, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	is_prime_Kernel << < liczba_blokow, liczba_watkow >> > (liczba, *czy_niepier, pierwiastek);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(&czy_niepierwsza_cuda, czy_niepier, sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaEventRecord(stop_counting, 0);
	cudaEventSynchronize(stop_counting);

	cudaEventElapsedTime(&computing_time, start_counting, stop_counting);

	cudaEventDestroy(start_counting);
	cudaEventDestroy(stop_counting);

	if (czy_niepierwsza_cuda)
	{
		std::cout << "Liczba " << liczba << " nie jest liczba pierwsza \n";
	}
	else
	{
		std::cout << "Liczba " << liczba << " jest liczba pierwsza \n";
	}
	printf("Czas wykonywania: %lu ms\n", computing_time);

Error:
	cudaFree(czy_niepier);
	return cudaStatus;
}