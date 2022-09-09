#include <cmath>
#include <vector>

extern "C" {

struct Vec2 {
    float x = 0.f;
    float y = 0.f;
    Vec2(): x(0), y(0) {}
    Vec2(float x, float y): x(x), y(y) {}
    Vec2(const Vec2& vec): x(vec.x), y(vec.y) {}

    void perp() {
        float tmp = x;
        x = -y;
        y = tmp;
    }

    Vec2 operator-(const Vec2& p) const {
        return Vec2(x - p.x, y - p.y);         // Return value optimized?
    }


    Vec2 operator+(const Vec2& p) const {
        return Vec2(x + p.x, y + p.y);         // Return value optimized?
    }

    Vec2 operator*(float val) const { 
        return Vec2(x * val, y * val);         // Return value optimized?
    }

    float get_angle() const {
        return atan2f(y, x);
    }

    float dot(const Vec2& p) const {
        return x * p.x + y * p.y;
    }

    float norm() const {
        return sqrtf(x * x + y * y);
    }
};

}