#pragma once

#ifndef NOMINMAX
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef byte

#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>
#include <cmath>

// Forward declaration of Neural_Network
struct Neural_Network;

class GUIMaker {
private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    Neural_Network* NN;
    HWND hwnd;
    std::thread guiThread;
    std::mutex mtx;
    std::condition_variable cv;
    bool running;
    std::wstring windowTitle;
    HDC memDC;
    HBITMAP memBitmap;

    void CreateGUIWindow();
    void GUIThreadFunction();
    void drawNeuralNetwork();

    static const int WINDOW_WIDTH = 1700;
    static const int WINDOW_HEIGHT = 900;

public:
    GUIMaker(const std::wstring& title, Neural_Network* neuralNetwork);
    ~GUIMaker();

    void Initialize();
    void Shutdown();
};

// Implementation

inline GUIMaker::GUIMaker(const std::wstring& title, Neural_Network* neuralNetwork)
    : NN(neuralNetwork), hwnd(nullptr), running(false), windowTitle(title), memDC(NULL), memBitmap(NULL) {}

inline GUIMaker::~GUIMaker() {
    Shutdown();
}

inline void GUIMaker::Initialize() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!running) {
        running = true;
        guiThread = std::thread(&GUIMaker::GUIThreadFunction, this);
    }
}

inline void GUIMaker::Shutdown() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        running = false;
    }
    cv.notify_one();
    if (guiThread.joinable()) {
        guiThread.join();
    }
    if (memDC) DeleteDC(memDC);
    if (memBitmap) DeleteObject(memBitmap);
}

inline void GUIMaker::GUIThreadFunction() {
    CreateGUIWindow();
    
    MSG msg = {};
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

inline void GUIMaker::CreateGUIWindow() {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"NeuralNetworkWindow";
    RegisterClassW(&wc);

    hwnd = CreateWindowExW(
        0, L"NeuralNetworkWindow", windowTitle.c_str(),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        WINDOW_WIDTH, WINDOW_HEIGHT, NULL, NULL, GetModuleHandleW(NULL), this
    );

    if (hwnd == NULL) {
        return;
    }

    SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

    HDC hdc = GetDC(hwnd);
    memDC = CreateCompatibleDC(hdc);
    memBitmap = CreateCompatibleBitmap(hdc, WINDOW_WIDTH, WINDOW_HEIGHT);
    SelectObject(memDC, memBitmap);
    ReleaseDC(hwnd, hdc);

    ShowWindow(hwnd, SW_SHOW);

    SetTimer(hwnd, 1, 1000 / 60, NULL);  // 60 FPS
}

inline void GUIMaker::drawNeuralNetwork() {
    // Fill background
    RECT clientRect = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
    FillRect(memDC, &clientRect, (HBRUSH)(COLOR_WINDOW + 1));

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

    int radius = 20;
    int buffer = 50;

    int x_division_width = (WINDOW_WIDTH - (2 * buffer)) / (NN->neuron_layers.size() - 1);

    std::vector<std::vector<POINT>> layerPoints;

    // First pass: Store neuron positions
    for (int layer = 0; layer < NN->neuron_layers.size(); layer++) {
        std::vector<POINT> currentLayerPoints;
        for (int neuron = 0; neuron < NN->neuron_layers[layer].rows(); neuron++) {
            int y_division_width = (WINDOW_HEIGHT - (2 * buffer)) / (NN->neuron_layers[layer].rows() - 1);

            // int x = centre_x - ((0.8 * WINDOW_WIDTH) / 2) + ((layer) * x_division_width);
            int x = buffer + (layer * x_division_width);
            int y = WINDOW_HEIGHT/2;
            
            if (y_division_width != 0) {
                // y = (int)(double)centre_y - ((0.9 * WINDOW_HEIGHT) / 2) + ((neuron) * y_division_width);
                y = buffer + (neuron * y_division_width);
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
            TextOutA(memDC, point.x - 15, point.y - 7, activationText, strlen(activationText));
        }
    }

    // Copy the off-screen buffer to the window
    HDC hdc = GetDC(hwnd);
    BitBlt(hdc, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, memDC, 0, 0, SRCCOPY);
    ReleaseDC(hwnd, hdc);
}

inline LRESULT CALLBACK GUIMaker::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    GUIMaker* pThis = reinterpret_cast<GUIMaker*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (uMsg) {
    case WM_PAINT:
        if (pThis) {
            pThis->drawNeuralNetwork();
        }
        return 0;
    case WM_TIMER:
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    case WM_DESTROY:
        KillTimer(hwnd, 1);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}