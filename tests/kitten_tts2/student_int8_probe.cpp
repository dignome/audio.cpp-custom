#include "engine/community_models/kitten_tts2/student_int8.h"
#include "engine/framework/core/execution_context.h"
#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
namespace k=engine::community_models::kitten_tts2;
namespace c=engine::core;
namespace a=engine::assets;
void check(c::ExecutionContext & execution, const k::StudentInt8Weights & w, int rows, double & worst) {
    std::mt19937 rng(1234); std::uniform_real_distribution<float> dist(-3,3);
    std::vector<float> input(rows*w.input);
    for(auto & x:input)x=dist(rng);
    for(int64_t j=0;j<w.input;++j)input[j]=0;
    if(rows>1) { for(int64_t j=0;j<w.input;++j)input[w.input+j]=(static_cast<int>(j%255)-127)*.5f;input[2*w.input-1]=127; }
    if(rows>2)for(int64_t j=0;j<w.input;++j)input[2*w.input+j]*=1e-8f;
    if(rows>3)for(int64_t j=0;j<w.input;++j)input[3*w.input+j]=3;
    if(rows>4)for(int64_t j=0;j<w.input;++j)input[4*w.input+j]=-3;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({2*1024*1024,nullptr,true}),ggml_free);
    c::ModuleBuildContext build{ctx.get(),"student-int8-test",c::BackendType::Cpu};
    auto x=c::make_tensor(build,GGML_TYPE_F32,c::TensorShape::from_dims({rows,w.input}));ggml_set_input(x.tensor);
    auto y=k::build_student_int8_linear(build,x,w);ggml_set_output(y.tensor);
    auto * graph=ggml_new_graph_custom(ctx.get(),64,false);ggml_build_forward_expand(graph,y.tensor);
    std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)> allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())),ggml_gallocr_free);
    if(!ggml_gallocr_alloc_graph(allocator.get(),graph))throw std::runtime_error("allocation failed");
    ggml_backend_tensor_set(x.tensor,input.data(),0,input.size()*4);
    if(c::compute_backend_graph(execution.backend(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute failed");
    auto actual=c::read_tensor_f32(y.tensor);
    double error=0,energy=0;
    for(int row=0;row<rows;++row) {
        float maximum=1e-5f;
        for(int64_t j=0;j<w.input;++j)maximum=std::max(maximum,std::abs(input[row*w.input+j]));
        float scale=127/maximum;
        std::vector<int> q(w.input);
        for(int64_t j=0;j<w.input;++j)q[j]=std::clamp(static_cast<int>(std::nearbyint(input[row*w.input+j]*scale)),-128,127);
        for(int64_t o=0;o<w.output;++o) {
            int32_t sum=0;double ref=0;
            for(int64_t j=0;j<w.input;++j) {
                int code=w.codes[o*w.stride+j];sum+=code*q[j];
                const float weight=static_cast<float>(code)/w.scales[o],activation=static_cast<float>(q[j])/scale;
                ref+=static_cast<double>(weight)*activation;
            }
            const float expected=(static_cast<float>(sum)/scale)/w.scales[o],got=actual[row*w.output+o];
            if(got!=expected)throw std::runtime_error("integer dot product/activation rounding mismatch");
            error+=(got-ref)*(got-ref);energy+=ref*ref;
        }
    }
    const double relative=std::sqrt(error/std::max(energy,1e-30));worst=std::max(worst,relative);
    if(relative>1e-6)throw std::runtime_error("linear reconstruction error exceeds tolerance");
    c::release_backend_graph_resources(execution.backend(),graph);
}
int main(int argc,char ** argv) {try {
    if(argc!=2)throw std::runtime_error("usage: kitten_tts2_student_int8_probe GGUF");
    c::BackendConfig config;config.type=c::BackendType::Cpu;config.threads=12;c::ExecutionContext execution(config);
    auto source=a::open_tensor_source(argv[1],"s3gen");size_t count=0,bytes=0;double worst=0;
    for(const auto & t:source->tensors()) {
        const std::string suffix=t.name.size()>=8 && t.name.substr(t.name.size()-8)==".packed8" ? ".packed8" : ".packed4";
        if(t.name.size()<suffix.size() || t.name.substr(t.name.size()-suffix.size())!=suffix)continue;
        auto name=t.name.substr(0,t.name.size()-suffix.size());auto shape=source->require_tensor_data(name+".shape");
        if(shape.bytes.size()!=16)continue;
        int64_t dims[2];std::memcpy(dims,shape.bytes.data(),16);
        auto w=k::load_student_int8_weights(*source,name,dims[1],dims[0]);check(execution,*w,7,worst);bytes+=w->codes.size();++count;
    }
    if(count != (source->has_tensor("flow.decoder.estimator.time_embed_mixer.weight.folded") ? 409 : 426))throw std::runtime_error("unexpected student linear count");
    k::StudentInt8Weights tail{17,7,32};tail.avx2=k::student_int8_has_avx2();tail.codes.resize(7*32,0);tail.sums.resize(7);tail.scales.resize(7);
    for(int o=0;o<7;++o) {tail.scales[o]=3.1f+o;for(int j=0;j<17;++j){int8_t q=static_cast<int8_t>((j+o)%16-8);tail.codes[o*32+j]=q;tail.sums[o]+=q;}}
    check(execution,tail,3,worst);tail.avx2=false;check(execution,tail,3,worst);
    tail.full_range=true;tail.avx2=k::student_int8_has_avx2();
    std::fill(tail.sums.begin(),tail.sums.end(),0);
    for(int o=0;o<7;++o)for(int j=0;j<17;++j) {
        int8_t q=static_cast<int8_t>(o==0?127:o==1?-128:((j+o)%2?127:-128));
        tail.codes[o*32+j]=q;tail.sums[o]+=q;
    }
    check(execution,tail,7,worst);tail.avx2=false;check(execution,tail,7,worst);
    std::cout<<"linears="<<count<<" padded_int8_bytes="<<bytes<<" avx2="<<k::student_int8_has_avx2()<<" worst_relative_l2="<<worst<<" scalar_and_tail_tests=passed\n";
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}
