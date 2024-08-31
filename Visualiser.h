#ifndef VISUALISER_H
#define VISUALISER_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdexcept>


class Win32GUI {
private:
    static const int WINDOW_WIDTH = 1000;
    static const int WINDOW_HEIGHT = 800;

    HWND hwnd;
    static Win32GUI* instance;
    HBITMAP memBitmap;
    HDC memDC;

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

public:
    Win32GUI();
    ~Win32GUI();

    void run();
    void redraw();

    template<typename T>
    void drawNeuralNetwork(T* NN) {
        // Helper function to map a value to a blue-red color
        auto valueToColor = [](double value) -> COLORREF {
            value = std::max(-1.0, std::min(1.0, value));  // Clamp value to [-1, 1]
            if (value < 0) {
                int intensity = (int)(-value * 255);
                return RGB(0, 0, intensity);  // Blue for negative
            } else {
                int intensity = (int)(value * 255);
                return RGB(intensity, 0, 0);  // Red for positive
            }
        };

        // Fill background
        RECT clientRect = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        FillRect(memDC, &clientRect, (HBRUSH)(COLOR_WINDOW + 1));

        int radius = 20;
        int centre_x = WINDOW_WIDTH/2;
        int centre_y = WINDOW_HEIGHT/2;

        int x_division_width = (double)(0.8 * WINDOW_WIDTH) / (NN->neuron_layers.size() + 1);

        std::vector<std::vector<POINT>> layerPoints;

        // First pass: Store neuron positions
        for (int layer = 0; layer < NN->neuron_layers.size(); layer++) {
            std::vector<POINT> currentLayerPoints;
            for (int neuron = 0; neuron < NN->neuron_layers[layer].rows(); neuron++) {
                int y_division_width = (double)(0.9 * WINDOW_HEIGHT) / (NN->neuron_layers[layer].rows() + 1);

                int x = centre_x - ((0.8 * WINDOW_WIDTH) / 2) + (layer + 1) * x_division_width;
                int y = centre_y;
                
                if (y_division_width != 0) {
                    y = centre_y - ((0.9 * WINDOW_HEIGHT) / 2) + (neuron + 1) * y_division_width;
                }

                currentLayerPoints.push_back({x, y});
            }
            layerPoints.push_back(currentLayerPoints);
        }

        // Second pass: Draw connections
        for (int layer = 0; layer < NN->neuron_layers.size() - 1; layer++) {
            for (int startNeuron = 0; startNeuron < layerPoints[layer].size(); startNeuron++) {
                const auto& startPoint = layerPoints[layer][startNeuron];
                for (int endNeuron = 0; endNeuron < layerPoints[layer + 1].size(); endNeuron++) {
                    const auto& endPoint = layerPoints[layer + 1][endNeuron];
                    
                    // Get the weight for this connection
                    double weight = NN->connection_layers[layer](endNeuron, startNeuron);
                    
                    // Map weight to color and thickness
                    COLORREF lineColor = valueToColor(weight);
                    int thickness = std::max(1, std::min(5, (int)(std::abs(weight) * 5)));  // Thickness between 1 and 5

                    HPEN connectionPen = CreatePen(PS_SOLID, thickness, lineColor);
                    HPEN oldPen = (HPEN)SelectObject(memDC, connectionPen);

                    MoveToEx(memDC, startPoint.x, startPoint.y, NULL);
                    LineTo(memDC, endPoint.x, endPoint.y);

                    SelectObject(memDC, oldPen);
                    DeleteObject(connectionPen);
                }
            }
        }

        // Third pass: Draw neurons
        for (int layer = 0; layer < layerPoints.size(); layer++) {
            for (int neuron = 0; neuron < layerPoints[layer].size(); neuron++) {
                const auto& point = layerPoints[layer][neuron];
                double activation = NN->neuron_layers[layer](neuron, 0);
                
                COLORREF neuronColor = valueToColor(activation);
                HBRUSH fillBrush = CreateSolidBrush(neuronColor);
                HBRUSH oldBrush = (HBRUSH)SelectObject(memDC, fillBrush);

                // Create a black pen for the outline
                HPEN outlinePen = CreatePen(PS_SOLID, 1, RGB(0, 0, 0));
                HPEN oldPen = (HPEN)SelectObject(memDC, outlinePen);

                // Draw the filled circle with an outline
                Ellipse(memDC, point.x - radius, point.y - radius, point.x + radius, point.y + radius);

                // Clean up
                SelectObject(memDC, oldBrush);
                SelectObject(memDC, oldPen);
                DeleteObject(fillBrush);
                DeleteObject(outlinePen);

                // Optionally, add activation value text
                char activationText[10];
                snprintf(activationText, sizeof(activationText), "%.2f", activation);
                SetBkMode(memDC, TRANSPARENT);
                SetTextColor(memDC, RGB(255, 255, 255)); // White text
                TextOut(memDC, point.x - 15, point.y - 7, activationText, strlen(activationText));
            }
        }

        // Force a redraw
        InvalidateRect(hwnd, NULL, FALSE);
    }

    // Getter for window dimensions
    static int getWindowWidth() { return WINDOW_WIDTH; }
    static int getWindowHeight() { return WINDOW_HEIGHT; }

private:
    void createMainWindow();
    void onPaint(HWND hwnd);
    void initializeMemoryDC();
};

// Implement the non-template member functions here


Win32GUI* Win32GUI::instance = nullptr;

Win32GUI::Win32GUI() : hwnd(nullptr), memBitmap(nullptr), memDC(nullptr) {
    instance = this;
    createMainWindow();
    initializeMemoryDC();
}

Win32GUI::~Win32GUI() {
    if (memDC) DeleteDC(memDC);
    if (memBitmap) DeleteObject(memBitmap);
    instance = nullptr;
}

void Win32GUI::run() {
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

void Win32GUI::redraw() {
    InvalidateRect(hwnd, nullptr, FALSE);
}

void Win32GUI::createMainWindow() {
    const char CLASS_NAME[] = "Win32 GUI Class";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        "Neural Network Visualizer",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT,
        nullptr,
        nullptr,
        GetModuleHandle(NULL),
        nullptr
    );

    if (hwnd == nullptr) {
        throw std::runtime_error("Failed to create window");
    }
}

void Win32GUI::initializeMemoryDC() {
    HDC hdc = GetDC(hwnd);
    memDC = CreateCompatibleDC(hdc);
    memBitmap = CreateCompatibleBitmap(hdc, WINDOW_WIDTH, WINDOW_HEIGHT);
    SelectObject(memDC, memBitmap);
    ReleaseDC(hwnd, hdc);
}

void Win32GUI::onPaint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);
    
    // Copy the memory DC to the window DC
    BitBlt(hdc, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, memDC, 0, 0, SRCCOPY);
    
    EndPaint(hwnd, &ps);
}

LRESULT CALLBACK Win32GUI::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_PAINT:
        if (instance) {
            instance->onPaint(hwnd);
        }
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

#endif