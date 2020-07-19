#ifndef RENDERING_UTILS_HPP
#define RENDERING_UTILS_HPP

#include "rendering.hpp"
#include "scene.hpp"
#include "script.hpp"
#include "utils.hpp"
#include <imgui.h>

struct Config_Item {
  enum Type { U32, F32, BOOL };
  Type type;
  union {
    u32  v_u32;
    f32  v_f32;
    bool v_bool;
  };
  union {
    u32 v_u32_min;
    u32 v_f32_min;
  };
  union {
    u32 v_u32_max;
    u32 v_f32_max;
  };
};

struct Tmp_List_Allocator {
  List *alloc() {
    List *out = (List *)tl_alloc_tmp(sizeof(List));
    memset(out, 0, sizeof(List));
    return out;
  }
};

struct Config {
  using string_t = inline_string<32>;
  Hash_Table<string_t, Config_Item> items;

  void init(string_ref init_script) {
    items.init();
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(init_script, Tmp_List_Allocator());
    traverse(cur);
  }

  void release() { items.release(); }

  void traverse(List *l) {
    struct Params {
      i32  imin = -1;
      f32  fmin = -1;
      i32  imax = -1;
      f32  fmax = -1;
      void traverse(List *l) {
        if (l == NULL) return;
        if (l->child) {
          traverse(l->child);
          traverse(l->next);
        } else {
          if (l->cmp_symbol("min")) {
            if (parse_decimal_int(l->get(1)->symbol.ptr, l->get(1)->symbol.len,
                                  &imin) == false)
              parse_float(l->get(1)->symbol.ptr, l->get(1)->symbol.len, &fmin);
          } else if (l->cmp_symbol("max")) {
            if (parse_decimal_int(l->get(1)->symbol.ptr, l->get(1)->symbol.len,
                                  &imax) == false)
              parse_float(l->get(1)->symbol.ptr, l->get(1)->symbol.len, &fmax);
          }
        }
      }
    };
    if (l == NULL) return;
    if (l->child) {
      traverse(l->child);
      traverse(l->next);
    } else {
      if (l->cmp_symbol("add")) {
        string_ref  type = l->get(1)->symbol;
        string_ref  name = l->get(2)->symbol;
        Config_Item item;
        MEMZERO(item);
        string_t _name;
        _name.init(name);
        if (type == stref_s("u32")) {
          Params params;
          params.traverse(l->get(4));
          item.type  = Config_Item::U32;
          item.v_u32 = l->get(3)->parse_int();
          if (params.imin != params.imax) {
            item.v_u32_min = params.imin;
            item.v_u32_max = params.imax;
          }
          items.insert(_name, item);
        } else if (type == stref_s("f32")) {
          Params params;
          params.traverse(l->get(4));
          if (params.fmin != params.fmax) {
            item.v_f32_min = params.fmin;
            item.v_f32_max = params.fmax;
          }
          item.type  = Config_Item::F32;
          item.v_f32 = l->get(3)->parse_float();
          items.insert(_name, item);
        } else if (type == stref_s("bool")) {
          item.type   = Config_Item::BOOL;
          item.v_bool = l->get(3)->parse_int() > 0;
          items.insert(_name, item);
        } else {
          TRAP;
        }
      }
    }
  }

  void on_imgui() {
    items.iter_pairs([&](string_t const &name, Config_Item &item) {
      char buf[0x100];
      snprintf(buf, sizeof(buf), "%.*s", STRF(name.ref()));
      if (item.type == Config_Item::U32) {
        if (item.v_u32_min != item.v_u32_max) {
          ImGui::SliderInt(buf, (int *)&item.v_u32, item.v_u32_min,
                           item.v_u32_max);
        } else {
          ImGui::InputInt(buf, (int *)&item.v_u32);
        }
      } else if (item.type == Config_Item::F32) {
        if (item.v_f32_min != item.v_f32_max) {
          ImGui::SliderFloat(buf, (float *)&item.v_f32, item.v_f32_min,
                             item.v_f32_max);
        } else {
          ImGui::InputFloat(buf, (float *)&item.v_f32);
        }
      } else if (item.type == Config_Item::BOOL) {
        ImGui::Checkbox(buf, &item.v_bool);
      } else {
        TRAP;
      }
    });
  }

  u32 &get_u32(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    ASSERT_DEBUG(items.contains(_name));
    return items.get_ref(_name).v_u32;
  }

  f32 &get_f32(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    ASSERT_DEBUG(items.contains(_name));
    return items.get_ref(_name).v_f32;
  }

  bool &get_bool(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    ASSERT_DEBUG(items.contains(_name));
    return items.get_ref(_name).v_bool;
  }

  void dump(FILE *file) {
    fprintf(file, "(config\n");
    items.iter_pairs([&](string_t const &name, Config_Item const &item) {
      if (item.type == Config_Item::U32) {
        if (item.v_u32_min != item.v_u32_max)
          fprintf(file, " (add u32 %.*s %i (min %i) (max %i))\n",
                  STRF(name.ref()), item.v_u32, item.v_u32_min, item.v_u32_max);
        else
          fprintf(file, " (add u32 %.*s %i)\n", STRF(name.ref()), item.v_u32);
      } else if (item.type == Config_Item::F32) {
        if (item.v_f32_min != item.v_f32_max)
          fprintf(file, " (add f32 %.*s %f (min %f) (max %f))\n",
                  STRF(name.ref()), item.v_f32, item.v_f32_min, item.v_f32_max);
        else
          fprintf(file, " (add f32 %.*s %f)\n", STRF(name.ref()), item.v_f32);
      } else if (item.type == Config_Item::BOOL) {
        fprintf(file, " (add bool %.*s %i)\n", STRF(name.ref()),
                item.v_bool ? 1 : 0);
      } else {
        TRAP;
      }
    });
    fprintf(file, ")\n");
  }
};

struct Camera {
  float  phi;
  float  theta;
  float  distance;
  float3 look_at;
  float  aspect;
  float  fov;
  float  znear;
  float  zfar;

  float3   pos;
  float4x4 view;
  float4x4 proj;
  float3   look;
  float3   right;
  float3   up;

  void init() {
    phi      = PI / 2.0f;
    theta    = PI / 2.0f;
    distance = 6.0f;
    look_at  = float3(0.0f, -4.0f, 0.0f);
    aspect   = 1.0;
    fov      = PI / 2.0;
    znear    = 1.0e-3f;
    zfar     = 10.0e5f;
  }

  void traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      traverse(l->child);
      traverse(l->next);
    } else {
      if (l->cmp_symbol("set_phi")) {
        phi = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_theta")) {
        theta = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_distance")) {
        distance = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_look_at")) {
        look_at.x = l->get(1)->parse_float();
        look_at.y = l->get(2)->parse_float();
        look_at.z = l->get(3)->parse_float();
      } else if (l->cmp_symbol("set_aspect")) {
        aspect = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_fov")) {
        fov = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_znear")) {
        znear = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_zfar")) {
        zfar = l->get(1)->parse_float();
      }
    }
  }

  void dump(FILE *file) {
    fprintf(file, "(camera\n");
    fprintf(file, " (set_phi %f)\n", phi);
    fprintf(file, " (set_theta %f)\n", theta);
    fprintf(file, " (set_distance %f)\n", distance);
    fprintf(file, " (set_look_at %f %f %f)\n", look_at.x, look_at.y, look_at.z);
    fprintf(file, " (set_aspect %f)\n", aspect);
    fprintf(file, " (set_fov %f)\n", fov);
    fprintf(file, " (set_znear %f)\n", znear);
    fprintf(file, " (set_zfar %f)\n", zfar);
    fprintf(file, ")\n");
  }

  void release() {}

  void update(float2 jitter = float2(0.0f, 0.0f)) {
    pos = float3(sinf(theta) * cosf(phi), cos(theta), sinf(theta) * sinf(phi)) *
              distance +
          look_at;
    look              = normalize(look_at - pos);
    right             = normalize(cross(look, float3(0.0f, 1.0f, 0.0f)));
    up                = normalize(cross(right, look));
    proj              = float4x4(0.0f);
    float tanHalfFovy = std::tan(fov * 0.5f);

    proj[0][0] = 1.0f / (aspect * tanHalfFovy);
    proj[1][1] = 1.0f / (tanHalfFovy);
    proj[2][2] = 0.0f;
    proj[2][3] = -1.0f;
    proj[3][2] = znear;

    proj[2][0] += jitter.x;
    proj[2][1] += jitter.x;
    view = glm::lookAt(pos, look_at, float3(0.0f, 1.0f, 0.0f));
  }
  float4x4 viewproj() { return proj * view; }
};

#endif // RENDERING_UTILS_HPP
