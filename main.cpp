#include <limits>
#include <iostream>
#include <chrono>
#include <cstdlib>
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

// Model -> World: W = Scale x Rotation x T
//     --             --
//     | Ux, Uy, Uz, 0 |
// T = | Vx, Vy, Vz, 0 |
//     | Wx, Wy, Wz, 0 |
//     | Ox, Oy, Oz, 1 |
//     --             --
// World -> Camera: V = Wc^(-1) = Tc^(-1) x Rc^(T)
//     --                        --
//     |    UCx,    UCy,   UCz, 0 |
// V = |    VCx,    VCy,   VCz, 0 |
//     |    WCx,    WCy,   WCz, 0 |
//     | -OC*UC, -OC*VC, OC*WC, 1 |
//     --                        --
extern mat<4,4> ModelView; // "OpenGL" state matrices
// Perspective Projection: (Clipping) -> Frustum -> Window
// -> NDC (normalized device coordinates) -> Normalize depth
//     --                           --
//     | 1/(r*tan(alpha/2)), 0, 0, 0 |
// P = | 0,     1/tan(alpha/2), 0, 0 |
//     | 0, 0,        f / (f - n), 1 |
//     | 0, 0,  - n * f / (f - n), 1 |
//     --                           --
// Notice: Z(z) = f/(f-n) - n*f/((f-n)*z), linear to 1/z
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
        // Lambert's Cosine Law: f(θ) = max(cosθ, 0) 
        //     = max(light_normal * surface_normal, 0)
        varing_intensity[nthvert] =
            std::max(0.0, model.normal(iface, nthvert) * normalize_light);
        // read the vertex from .obj file
        vec4 gl_Vertex = embed<4>(model.vert(iface, nthvert));
        // transform it to screen coordinates
        return Projection * ModelView * gl_Vertex;
    }

    virtual bool fragment(const vec3 bar, TGAColor& color) {
        // interpolate intensity for the current pixel
        float intensity = (varing_intensity * bar);
        // well duh
        color = TGAColor(255, 255, 255) * intensity;
        // do not discard this pixel
        return false;
    }
};

// Normalmapping
// Global (Cartesian) coordinate: interpret RGB as xyz.
// Darboux frame (i.e. tangent space):
//     z: normal to the object [B], (u x v / || u x v ||)
//     x: principal curvature direction, (xA * z(A^-1)^T = 0) (from uv: A^(-1) * (u1 - u0, u2 - u0, 0))
//     y: their cross product (from uv: A^(-1) * (v1 - v0, v2 - v0, 0), for a triangle, A = (p1- p0, p2 - p0, n))
struct TextureShader : IShader {
    const Model &model;
    vec3 l;               // light direction in normalized device coordinates
    mat<2,3> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    mat<3,3> varying_nrm; // normal per vertex to be interpolated by FS
    mat<3,3> ndc_tri;     // triangle in normalized device coordinates

    TextureShader(const Model &m) : model(m) {
        l = proj<3>((Projection*ModelView*embed<4>(light_dir, 0.))).normalize(); // transform the light vector to the normalized device coordinates
    }

    virtual vec4 vertex(const int iface, const int nthvert) {
        varying_uv.set_col(nthvert, model.uv(iface, nthvert));
        varying_nrm.set_col(nthvert, proj<3>((Projection*ModelView).invert_transpose()*embed<4>(model.normal(iface, nthvert), 0.)));
        vec4 gl_Vertex = Projection*ModelView*embed<4>(model.vert(iface, nthvert));
        ndc_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }

    // Phong's approximation of lighting model:
    //     Reflection = weight_A * Ambient (constant)
    //         + weight_D * Diffuse (cos(normal, light)) 
    //         + weight_S * Specular 
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
        // relection vector: r = l - 2(n * l)*n
        vec3 r = (n*(n*l)*2 - l).normalize(); // reflected light direction, specular mapping is described here: https://github.com/ssloy/tinyrenderer/wiki/Lesson-6-Shaders-for-the-software-renderer
        double spec = std::pow(std::max(r.z, 0.), 5+model.specular(uv)); // specular intensity, note that the camera lies on the z-axis (in ndc), therefore simple r.z

        TGAColor c = model.diffuse(uv);
        for (int i=0; i<3; i++)
            color[i] = std::min<int>(10 + c[i]*(diff + spec), 255); // (a bit of ambient light, diff + spec), clamp the result

        return false; // the pixel is not discarded
    }
};

// Shadow Mapping
struct DepthShader : public IShader {
    const Model &model;
    mat<3, 3> varing_tri;

    DepthShader(const Model &m) : model(m) {
    }
    
    virtual vec4 vertex(const int iface, const int nthvert) {
        // read the vertex from .obj file
        vec4 gl_Vertex = embed<4>(model.vert(iface, nthvert));
        // transform it to screen coordinates
        gl_Vertex = Projection * ModelView * gl_Vertex;
        varing_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }
    
    virtual bool fragment(const vec3 bar, TGAColor &color) {
        vec3 p = varing_tri * bar;
        color = TGAColor(255, 255, 255) * p.z;
        return false;
    }
};

struct ShadowShader : public IShader {
    const Model &model;
    std::vector<double> shadowbuffer;
    mat<4,4> uniform_M;   //  Projection*ModelView
    mat<4,4> uniform_MIT; // (Projection*ModelView).invert_transpose()
    mat<4,4> uniform_Mshadow; // transform framebuffer screen coordinates to shadowbuffer screen coordinates
    mat<2,3> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    mat<3,3> varying_tri; // triangle coordinates before Viewport transform, written by VS, read by FS

    ShadowShader(const Model &m, const std::vector<double> shadow,
                 const mat<4, 4> shadowM)
        : model(m), shadowbuffer(shadow)
    {
        uniform_M = ModelView;
        uniform_MIT = (Projection*ModelView).invert_transpose();
        uniform_Mshadow = shadowM * (Viewport * Projection * ModelView).invert();
    }

    virtual vec4 vertex(const int iface, const int nthvert) {
        varying_uv.set_col(nthvert, model.uv(iface, nthvert));
        vec4 gl_Vertex =
            Projection * ModelView * embed<4>(model.vert(iface, nthvert));
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }
    
    virtual bool fragment(const vec3 bar, TGAColor &color) {
        // corresponding point in the shadow buffer
        vec4 sb_p = uniform_Mshadow * embed<4>(varying_tri * bar);
        sb_p = sb_p / sb_p[3];
        // index in the shadowbuffer array
        int idx = int(sb_p[0]) + int(sb_p[1]) * width;
        float shadow = .3 + .7 * (shadowbuffer[idx] < sb_p[2]);
        // magic coeff to avoid z-fighting
        // float shadow =
        //     0.3f + 0.7f * (shadowbuffer[idx] < sb_p[2] + 43.34);

        // interpolate uv for the current pixel
        vec2 uv = varying_uv * bar;
        vec3 normal = proj<3>(uniform_MIT * embed<4>(model.normal(uv))).normalize();
        vec3 lit = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        vec3 reflect = (normal * (normal * lit * 2.0f) - lit).normalize();
        float spec = pow(std::max(reflect.z, 0.0), model.specular(uv));
        float diff = std::max(0.0, normal * lit);
        TGAColor c = model.diffuse(uv);
        for (int i = 0; i < 3; ++i)
            color[i] = std::min<float>(20 + c[i] * shadow * (1.2 * diff + 0.6 * spec), 255);
        return false;
    }
};

// Screen Space Ambient Occlusion
struct AOShader : public IShader {
    const Model &model;
    mat<4, 3> varing_tri;

    AOShader(const Model &m) : model(m) {}

    virtual vec4 vertex(const int iface, const int nthvert) {
        // read the vertex from .obj file
        vec4 gl_Vertex = embed<4>(model.vert(iface, nthvert));
        // transform it to screen coordinates
        gl_Vertex = Projection * ModelView * gl_Vertex;
        varing_tri.set_col(nthvert, gl_Vertex);
        return gl_Vertex;
    }
    
    virtual bool fragment(const vec3 /*bar*/, TGAColor &color) {
        color = TGAColor(0, 0, 0);
        return false;
    }

    static float max_elevation_angle(std::vector<double> zbuffer, vec2 p, vec2 dir) {
        float maxangle = 0;
        for (float t = 0; t < 1000; t += 1.0f) {
            vec2 cur = p + dir * t;
            if (cur.x >= width || cur.y >= height || cur.x < 0 || cur.y < 0)
                return maxangle;
            float distance = (p - cur).norm();
            if (distance < 1.0f)
                continue;
            float elevation = zbuffer[int(cur.x) + int(cur.y) * width]
                - zbuffer[int(p.x) + int(p.y)* width];
            maxangle = std::max(maxangle, atanf(elevation / distance));
        }
        return maxangle;
    }
};

enum RenderType {
    Wireframe = 0,
    Gouraud = 1,
    Textured = 2,
    Shadow = 3,
    SSAO = 4
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
        RenderType type = Shadow;
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
        {
            Model model(argv[m]);
            TextureShader shader(model);
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
        case Shadow:
        {
            Model model(argv[m]);

            // Pass1
            TGAImage depth(width, height, TGAImage::RGB);
            std::vector<double> shadowbuffer(
                width * height, -std::numeric_limits<double>::max());
            lookat(light_dir, center, up);
            viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
            projection(0);
            mat<4, 4> shadowMatrix = Viewport * Projection * ModelView;

            DepthShader depth_shader(model);
            for (int i = 0; i < model.nfaces(); i++) {
                vec4 clip_vert[3];
                for (int j = 0; j < 3; j++)
                    clip_vert[j] = depth_shader.vertex(i, j);
                triangle(clip_vert, depth_shader, depth, shadowbuffer);
            }

            // output depth image
            std::string result_name = argv[m];
            size_t lastindex = result_name.find_last_of(".");
            result_name = result_name.substr(0, lastindex) + "_depth.tga";
            depth.write_tga_file(result_name.c_str());
            std::cout << "Output Depth Image to: " << result_name << std::endl;

            // Pass2
            // Update matrix
            lookat(eye, center, up);
            viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
            projection(-1.f / (eye - center).norm());
            ShadowShader shadow_shader(model, shadowbuffer, shadowMatrix);
            for (int i = 0; i < model.nfaces(); i++) {
                vec4 clip_vert[3];
                for (int j = 0; j < 3; j++)
                    clip_vert[j] = shadow_shader.vertex(i, j);
                triangle(clip_vert, shadow_shader, framebuffer, zbuffer);
            }
            break;
        }
        case SSAO:
        default:
        {
            Model model(argv[m]);
            AOShader shader(model);
            for (int i = 0; i < model.nfaces(); i++) {
                vec4 clip_vert[3];
                for (int j = 0; j < 3; j++)
                    clip_vert[j] = shader.vertex(i, j);
                triangle(clip_vert, shader, framebuffer, zbuffer);
            }

            for(int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    if(zbuffer[x + y * width] < -1e5)
                        continue;
                    float total = 0;
                    for(float a = 0; a < M_PI * 2-1e-4 ; a += M_PI_4)
                        total += M_PI_2 - AOShader::max_elevation_angle(zbuffer, 
                            vec2(x, y), vec2(cos(a), sin(a)));
                    total /= (M_PI * 4);
                    total = pow(total, 100.f);
                    framebuffer.set(x, y, TGAColor(total * 255, total * 255, total * 255));
                }
            }
            break;
        }
        }
        // the vertical flip is moved inside the function
        std::string result_name = argv[m];
        size_t lastindex = result_name.find_last_of(".");
        result_name = result_name.substr(0, lastindex) + "_framebuffer.tga";
        framebuffer.write_tga_file(result_name.c_str());
        std::cout << "Output to: " << result_name << std::endl;
    }
    return 0;
}
