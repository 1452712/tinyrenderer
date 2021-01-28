#include <limits>
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include "model.h"
#include "our_gl.h"

constexpr int width  = 800; // output image size
constexpr int height = 800;

const vec3 light_dir(1,1,1); // light source
const vec3       eye(1,1,3); // camera position
const vec3    center(0,0,0); // camera direction
const vec3        up(0,1,0); // camera up vector

const TGAColor WHITE = TGAColor(255, 255, 255, 255);
const TGAColor RED = TGAColor(255, 0, 0, 255);

extern mat<4,4> ModelView; // "OpenGL" state matrices
extern mat<4,4> Projection;
extern mat<4,4> Viewport;

struct GouraudShader : public IShader {
    const Model &model;
    vec3 normalize_light;
    vec3 varing_intensity; // written by vertex shader, read by fragment shader

    GouraudShader(const Model &m) : model(m) { 
        normalize_light = light_dir;
        normalize_light.normalize();
    }

    virtual vec4 vertex(int iface, int nthvert) {
        // get diffuse lighting intensity
        varing_intensity[nthvert] =
            std::max(0.0, model.normal(iface, nthvert) * normalize_light);
        // read the vertex from .obj file
        vec4 gl_Vertex = embed<4>(model.vert(iface, nthvert));
        // transform it to screen coordinates
        return Viewport * Projection * ModelView * gl_Vertex;
    }

    virtual bool fragment(const vec3 bar, TGAColor& color) {
        // interpolate intensity for the current pixel
        float intensity = varing_intensity * bar;
        // well duh
        color = TGAColor(255, 255, 255) * intensity;
        // do not discard this pixel
        return false;
    }
};

struct Shader : IShader {
    const Model &model;
    vec3 l;               // light direction in normalized device coordinates
    mat<2,3> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    mat<3,3> varying_nrm; // normal per vertex to be interpolated by FS
    mat<3,3> ndc_tri;     // triangle in normalized device coordinates

    Shader(const Model &m) : model(m) {
        l = proj<3>((Projection*ModelView*embed<4>(light_dir, 0.))).normalize(); // transform the light vector to the normalized device coordinates
    }

    virtual vec4 vertex(const int iface, const int nthvert) {
        varying_uv.set_col(nthvert, model.uv(iface, nthvert));
        varying_nrm.set_col(nthvert, proj<3>((Projection*ModelView).invert_transpose()*embed<4>(model.normal(iface, nthvert), 0.)));
        vec4 gl_Vertex = Projection*ModelView*embed<4>(model.vert(iface, nthvert));
        ndc_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(const vec3 bar, TGAColor &color) {
        vec3 bn = (varying_nrm*bar).normalize(); // per-vertex normal interpolation
        vec2 uv = varying_uv*bar; // tex coord interpolation

        // for the math refer to the tangent space normal mapping lecture
        // https://github.com/ssloy/tinyrenderer/wiki/Lesson-6bis-tangent-space-normal-mapping
        mat<3,3> AI = mat<3,3>{ {ndc_tri.col(1) - ndc_tri.col(0), ndc_tri.col(2) - ndc_tri.col(0), bn} }.invert();
        vec3 i = AI * vec3(varying_uv[0][1] - varying_uv[0][0], varying_uv[0][2] - varying_uv[0][0], 0);
        vec3 j = AI * vec3(varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0);
        mat<3,3> B = mat<3,3>{ {i.normalize(), j.normalize(), bn} }.transpose();

        vec3 n = (B * model.normal(uv)).normalize(); // transform the normal from the texture to the tangent space

        double diff = std::max(0., n*l); // diffuse light intensity
        vec3 r = (n*(n*l)*2 - l).normalize(); // reflected light direction, specular mapping is described here: https://github.com/ssloy/tinyrenderer/wiki/Lesson-6-Shaders-for-the-software-renderer
        double spec = std::pow(std::max(r.z, 0.), 5+model.specular(uv)); // specular intensity, note that the camera lies on the z-axis (in ndc), therefore simple r.z

        TGAColor c = model.diffuse(uv);
        for (int i=0; i<3; i++)
            color[i] = std::min<int>(10 + c[i]*(diff + spec), 255); // (a bit of ambient light, diff + spec), clamp the result

        return false; // the pixel is not discarded
    }
};

enum RenderType {
    Wireframe = 0,
    Gouraud = 1,
    Textured = 2
};

int main(int argc, char** argv)
{
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }

    std::vector<double> zbuffer(width*height, -std::numeric_limits<double>::max()); // note that the z-buffer is initialized with minimal possible values
    TGAImage framebuffer(width, height, TGAImage::RGB); // the output image
    lookat(eye, center, up);                            // build the ModelView matrix
    viewport(width/8, height/8, width*3/4, height*3/4); // build the Viewport matrix
    projection(-1.f/(eye-center).norm());               // build the Projection matrix

    for (int m=1; m<argc; m++) { // iterate through all input objects
        RenderType type = Gouraud;
        switch(type) {
        case Wireframe:
        {
            Model model(argv[m]);
            for (int i = 0; i < model.nfaces(); i++) {
                for (int j = 0; j < 3; j++) {
                    vec3 v0 = model.vert(i, j);
                    vec3 v1 = model.vert(i, (j + 1) % 3);
                    int x0 = (v0.x + 1.) * width / 2.;
                    int y0 = (v0.y + 1.) * height / 2.;
                    int x1 = (v1.x + 1.) * width / 2.;
                    int y1 = (v1.y + 1.) * height / 2.;
                    line(x0, y0, x1, y1, framebuffer, WHITE);
                }
            }
            break;
        }
        case Gouraud:
        {
            Model model(argv[m]);
            GouraudShader shader(model);
            for (int i = 0; i < model.nfaces(); i++) {
                vec4 clip_vert[3];
                for (int j = 0; j < 3; j++)
                    clip_vert[j] = shader.vertex(i, j);
                triangle(clip_vert, shader, framebuffer, zbuffer);
            }
            break;
        }
        case Textured:
        default:
        {
            Model model(argv[m]);
            Shader shader(model);
            for (int i = 0; i < model.nfaces(); i++) {                      // for every triangle
                vec4 clip_vert[3]; // triangle coordinates (clip coordinates),
                                   // written by VS, read by FS
                for (int j = 0; j < 3; j++)
                    clip_vert[j] = shader.vertex(
                        i,
                        j); // call the vertex shader for each triangle vertex
                triangle(clip_vert, shader, framebuffer,
                         zbuffer); // actual rasterization routine call
            }
            break;
        }
        }
        // the vertical flip is moved inside the function
        std::string result_name = argv[m];
        size_t lastindex = result_name.find_last_of(".");
        result_name = result_name.substr(0, lastindex) + "_framebuffer.tga";
        framebuffer.write_tga_file(result_name.c_str());
        std::cout << "Output to " << result_name << std::endl;
    }
    return 0;
}
