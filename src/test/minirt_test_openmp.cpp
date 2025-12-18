#include "minirt/minirt.h"
#include <cmath>
#include <iostream>
#include <iomanip>  // Добавьте этот заголовочный файл
#include <omp.h>
#include <chrono>
#include <vector>   // Добавьте для std::vector
#include <string> 

using namespace minirt;
using namespace std::chrono;

void initScene(Scene &scene) {
    Color red {1, 0.2, 0.2};
    Color blue {0.2, 0.2, 1};
    Color green {0.2, 1, 0.2};
    Color white {0.8, 0.8, 0.8};
    Color yellow {1, 1, 0.2};

    Material metallicRed {red, white, 50};
    Material mirrorBlack {Color {0.0}, Color {0.9}, 1000};
    Material matteWhite {Color {0.7}, Color {0.3}, 1};
    Material metallicYellow {yellow, white, 250};
    Material greenishGreen {green, 0.5, 0.5};

    Material transparentGreen {green, 0.8, 0.2};
    transparentGreen.makeTransparent(1.0, 1.03);
    Material transparentBlue {blue, 0.4, 0.6};
    transparentBlue.makeTransparent(0.9, 0.7);

    scene.addSphere(Sphere {{0, -2, 7}, 1, transparentBlue});
    scene.addSphere(Sphere {{-3, 2, 11}, 2, metallicRed});
    scene.addSphere(Sphere {{0, 2, 8}, 1, mirrorBlack});
    scene.addSphere(Sphere {{1.5, -0.5, 7}, 1, transparentGreen});
    scene.addSphere(Sphere {{-2, -1, 6}, 0.7, metallicYellow});
    scene.addSphere(Sphere {{2.2, 0.5, 9}, 1.2, matteWhite});
    scene.addSphere(Sphere {{4, -1, 10}, 0.7, metallicRed});

    scene.addLight(PointLight {{-15, 0, -15}, white});
    scene.addLight(PointLight {{1, 1, 0}, blue});
    scene.addLight(PointLight {{0, -10, 6}, red});

    scene.setBackground({0.05, 0.05, 0.08});
    scene.setAmbient({0.1, 0.1, 0.1});
    scene.setRecursionLimit(20);

    scene.setCamera(Camera {{0, 0, -20}, {0, 0, 0}});
}


int main(int argc, char** argv) {
    int viewPlaneResolutionX = (argc > 1 ? std::stoi(argv[1]) : 2000);
    int viewPlaneResolutionY = (argc > 2 ? std::stoi(argv[2]) : 2000);
    int numOfSamples = (argc > 3 ? std::stoi(argv[3]) : 1);
    std::string sceneFile = (argc > 4 ? argv[4] : "");

    Scene scene;
    if (sceneFile.empty()) {
        initScene(scene);
    }
    else {
        scene.loadFromFile(sceneFile);
    }

    const double backgroundSizeX = 4;
    const double backgroundSizeY = 4;
    const double backgroundDistance = 15;
    const double viewPlaneDistance = 5;
    const double viewPlaneSizeX = backgroundSizeX * viewPlaneDistance / backgroundDistance;
    const double viewPlaneSizeY = backgroundSizeY * viewPlaneDistance / backgroundDistance;

    int max_threads = omp_get_max_threads();

    std::cout << "Image: " << viewPlaneResolutionX << "x" << viewPlaneResolutionY << std::endl;
    std::cout << "Max available threads: " << max_threads << std::endl;
    Image image(viewPlaneResolutionX, viewPlaneResolutionY);
    ViewPlane viewPlane{ viewPlaneResolutionX, viewPlaneResolutionY,
                         viewPlaneSizeX, viewPlaneSizeY, viewPlaneDistance };
    std::vector<std::string> schedule_types = { "static", "dynamic", "guided" };

    std::vector<int> chunk_sizes = { 1, 5, 10, 50, 100};

    std::cout << "PART 1: WITHOUT COLLAPSE (NESTED LOOPS)" << std::endl;
    for (const auto& schedule_type : schedule_types) {
        std::cout << "\n[" << schedule_type << " SCHEDULE]" << std::endl;
        for (int chunk_size : chunk_sizes) {
            std::cout << "\nChunk size: " << chunk_size << std::endl;

            for (int num_threads = 1; num_threads < max_threads; num_threads++) {
                omp_set_num_threads(num_threads);

                double start_time = omp_get_wtime();

                if (schedule_type == "static") {
#pragma omp parallel for schedule(static, chunk_size)
                    for (int x = 0; x < viewPlaneResolutionX; x++) {
                        for (int y = 0; y < viewPlaneResolutionY; y++) {
                            image.set(x, y, viewPlane.computePixel(scene, x, y, numOfSamples));
                        }
                    }
                }
                else if (schedule_type == "dynamic") {
#pragma omp parallel for schedule(dynamic, chunk_size)
                    for (int x = 0; x < viewPlaneResolutionX; x++) {
                        for (int y = 0; y < viewPlaneResolutionY; y++) {
                            image.set(x, y, viewPlane.computePixel(scene, x, y, numOfSamples));
                        }
                    }
                }
                else if (schedule_type == "guided") {
#pragma omp parallel for schedule(guided, chunk_size)
                    for (int x = 0; x < viewPlaneResolutionX; x++) {
                        for (int y = 0; y < viewPlaneResolutionY; y++) {
                            image.set(x, y, viewPlane.computePixel(scene, x, y, numOfSamples));
                        }
                    }
                }

                double end_time = omp_get_wtime();
                double elapsed_time = end_time - start_time;

                std::cout << "  Threads: " << std::setw(2) << num_threads
                    << "  Time: " << std::fixed << std::setprecision(3)
                    << elapsed_time << " s"
                    << "  Speed: " << std::setprecision(2) << std::endl;
            }
        }
    }

    std::cout << "PART 2: WITH COLLAPSE(2)" << std::endl;
    for (const auto& schedule_type : schedule_types) {
        std::cout << "\n[" << schedule_type << " SCHEDULE with collapse(2)]" << std::endl;

        for (int chunk_size : chunk_sizes) {
            std::cout << "\nChunk size: " << chunk_size << std::endl;
            for (int num_threads = 1; num_threads < max_threads; num_threads++) {

                omp_set_num_threads(num_threads);

                double start_time = omp_get_wtime();

                if (schedule_type == "static") {
#pragma omp parallel for collapse(2) schedule(static, chunk_size)
                    for (int x = 0; x < viewPlaneResolutionX; x++) {
                        for (int y = 0; y < viewPlaneResolutionY; y++) {
                            image.set(x, y, viewPlane.computePixel(scene, x, y, numOfSamples));
                        }
                    }
                }
                else if (schedule_type == "dynamic") {
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
                    for (int x = 0; x < viewPlaneResolutionX; x++) {
                        for (int y = 0; y < viewPlaneResolutionY; y++) {
                            image.set(x, y, viewPlane.computePixel(scene, x, y, numOfSamples));
                        }
                    }
                }
                else if (schedule_type == "guided") {
#pragma omp parallel for collapse(2) schedule(guided, chunk_size)
                    for (int x = 0; x < viewPlaneResolutionX; x++) {
                        for (int y = 0; y < viewPlaneResolutionY; y++) {
                            image.set(x, y, viewPlane.computePixel(scene, x, y, numOfSamples));
                        }
                    }
                }

                double end_time = omp_get_wtime();
                double elapsed_time = end_time - start_time;

                std::cout << "  Threads: " << std::setw(2) << num_threads
                    << "  Time: " << std::fixed << std::setprecision(3)
                    << elapsed_time << " s"
                    << "  Speed: " << std::setprecision(2) << std::endl;
            }
        }
    }
    image.saveJPEG("REALLY.jpg");
    return 0;
}