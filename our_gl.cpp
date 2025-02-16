#include <limits>
#include "our_gl.h"

// draw v on the screen:
// Viewport * Projection * View * Model * v
mat<4,4> ModelView;
mat<4,4> Viewport;
mat<4,4> Projection;

void viewport(const int x, const int y, const int w, const int h) {
    // point belongs to [[-1, 1], [-1, 1]]
    // (v.x+1)/2 scale to [0, 1]
    // (v.x+1)*width/2 sweeps all the image

    // bi-unit cube [-1,1]*[-1,1]*[-1,1] is mapped onto 
    // the screen cube [x,x+w]*[y,y+h]*[0,d]
    Viewport = {{{w/2., 0, 0, x+w/2.}, {0, h/2., 0, y+h/2.}, {0,0,1,0}, {0,0,0,1}}};
}

void projection(const double coeff) { // check https://github.com/ssloy/tinyrenderer/wiki/Lesson-4-Perspective-projection
    Projection = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,coeff,1}}};
}

void lookat(const vec3 eye, const vec3 center, const vec3 up) { // check https://github.com/ssloy/tinyrenderer/wiki/Lesson-5-Moving-the-camera
    vec3 z = (eye-center).normalize();
    vec3 x =  cross(up,z).normalize();
    vec3 y =  cross(z, x).normalize();
    mat<4,4> Minv = {{{x.x,x.y,x.z,0},   {y.x,y.y,y.z,0},   {z.x,z.y,z.z,0},   {0,0,0,1}}};
    mat<4,4> Tr   = {{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
    ModelView = Minv*Tr;
}

// P = (1-u-v)A + uB + vC
vec3 barycentric(const vec2 tri[3], const vec2 P) {
    mat<3,3> ABC = {{embed<3>(tri[0]), embed<3>(tri[1]), embed<3>(tri[2])}};
    if (ABC.det()<1e-3) return vec3(-1,1,1); // for a degenerate triangle generate negative coordinates, it will be thrown away by the rasterizator
    return ABC.invert_transpose() * embed<3>(P);
}

void triangle(const vec4 clip_verts[3], IShader &shader, TGAImage &image, std::vector<double> &zbuffer) {
    // using homogeneous coord to proj objects onto screen
    vec4 pts[3]  = { Viewport*clip_verts[0],    Viewport*clip_verts[1],    Viewport*clip_verts[2]    };  // triangle screen coordinates before persp. division
    vec2 pts2[3] = { proj<2>(pts[0]/pts[0][3]), proj<2>(pts[1]/pts[1][3]), proj<2>(pts[2]/pts[2][3]) };  // triangle screen coordinates after  perps. division

    vec2 bboxmin( std::numeric_limits<double>::max(),  std::numeric_limits<double>::max());
    vec2 bboxmax(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());
    vec2 clamp(image.get_width()-1, image.get_height()-1);
    for (int i=0; i<3; i++)
        for (int j=0; j<2; j++) {
            bboxmin[j] = std::max(0.,       std::min(bboxmin[j], pts2[i][j]));
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts2[i][j]));
        }
#pragma omp parallel for
    for (int x=(int)bboxmin.x; x<=(int)bboxmax.x; x++) {
        for (int y=(int)bboxmin.y; y<=(int)bboxmax.y; y++) {
            vec3 bc_screen  = barycentric(pts2, vec2(x, y));
            vec3 bc_clip    = vec3(bc_screen.x/pts[0][3], bc_screen.y/pts[1][3], bc_screen.z/pts[2][3]);
            bc_clip = bc_clip/(bc_clip.x+bc_clip.y+bc_clip.z); // check https://github.com/ssloy/tinyrenderer/wiki/Technical-difficulties-linear-interpolation-with-perspective-deformations
            double frag_depth = vec3(clip_verts[0][2], clip_verts[1][2], clip_verts[2][2])*bc_clip;
            // backface culling & hidden face removal
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0 || zbuffer[x+y*image.get_width()]>frag_depth) continue;
            TGAColor color;
            bool discard = shader.fragment(bc_clip, color);
            if (discard) continue;
            // 0: farest, 1: closest
            zbuffer[x+y*image.get_width()] = frag_depth;
            image.set(x, y, color);
        }
    }
}

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    bool steep = false;
    if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    const int dx = x1 - x0;
    const int dy = y1 - y0;
    const float derror = std::abs(dy) * 2;
    float error = 0;
    int y = y0;
    for(int x = x0; x <= x1; x++) {
        if (steep) {
            image.set(y, x, color);
        } else {
            image.set(x, y, color);
        }
        error += derror;
        if (error > dx) {
            y += (y1 > y0)? 1 : -1;
            error -= dx * 2;
        }
    }
}
