#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/util/channel.hpp"
#include "caffe/util/mpi_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  param.mutable_state()->set_phase(phase);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG(INFO) << "Initializing net from parameters: " << std::endl
            << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  CHECK(param.input_dim_size() == 0 || param.input_shape_size() == 0)
      << "Must specify either input_shape OR deprecated input_dim, not both.";
  if (param.input_dim_size() > 0) {
    // Deprecated 4D dimensions.
    CHECK_EQ(param.input_size() * 4, param.input_dim_size())
        << "Incorrect input blob dimension specifications.";
  } else {
    CHECK_EQ(param.input_size(), param.input_shape_size())
        << "Exactly one input_shape must be specified per input.";
  }
  memory_used_ = 0;
  // set the input blobs
  for (int input_id = 0; input_id < param.input_size(); ++input_id) {
    const int layer_id = -1;  // inputs have fake layer ID -1
    AppendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);

    // input blobs are excluded from memory optimization by default
    excluded_blob_names_.insert(param.input(input_id));
  }
  DLOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());

  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }

    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    LOG(INFO) << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    #ifdef USE_MPI
    vector<bool> source_layer_need_sync;
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {

      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      int src_layer_id = top_layer_indices_[blob_id].first;
      if (src_layer_id>=0) source_layer_need_sync.push_back(layers_[src_layer_id]->need_sync());
      if (source_layer_need_sync.size()>0){
        CHECK_EQ(source_layer_need_sync.back(), source_layer_need_sync[0])
          <<" blob "<<layer_param.bottom(0)
          <<" and blob "<< layer_param.bottom(bottom_id)
          <<" are from layers with different paralle mode. This is not supported.";
      }
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }

    if (layers_[layer_id]->is_gathering()){
      layers_[layer_id]->set_need_sync(false);
    } else {
      if(layers_[layer_id]->is_scattering()){
        layers_[layer_id]->set_need_sync(true);
      } else {
        if ((source_layer_need_sync.size() > 0)) {
          layers_[layer_id]->set_need_sync(source_layer_need_sync[0]);
          LOG(INFO) << "This layer is inheriting previous layer's sync mode: " << source_layer_need_sync[0];
        }
      }
    }
    #else
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    #endif

    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    LOG(INFO) << "Setting up " << layer_names_[layer_id];
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG(INFO) << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG(INFO) << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    DLOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() > 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;

        //special treatment for "Gather" layer
        //This layer should be transparent to bp inferring.
        if (strcmp(layers_[layer_id]->type(), "Gather")==0){
          blob_need_backward_[top_id_vecs_[layer_id][top_id]]
              = blob_need_backward_[bottom_id_vecs_[layer_id][top_id]];
        }
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (layer_need_backward_[layer_id]) {
      LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
    } else {
      LOG(INFO) << layer_names_[layer_id]
                << " does not need backward computation.";
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG(INFO) << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);

    // add output blob name to default excluded blobs
    excluded_blob_names_.insert(*it);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  GetLearningRateAndWeightDecay();
  debug_info_ = param.debug_info();
  LOG(INFO) << "Network initialization done.";
  LOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);

  // optimize memory
  optimize_memory_ = (param.mem_param().optimize_train() && phase_ == TRAIN) ||
                     (param.mem_param().optimize_test() && phase_ == TEST);

  // add additional specified blobs to the exclusion list
  for (int ex_id = 0; ex_id < param.mem_param().exclude_blob_size(); ++ex_id){
    excluded_blob_names_.insert(param.mem_param().exclude_blob(ex_id));
  }

  // launch memory optimization if necessary
  if (!debug_info_ && optimize_memory_) {
    MemoryOptimize_v2();
  }
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG(INFO) << "The NetState phase (" << state.phase()
          << ") differed from the phase (" << rule.phase()
          << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG(INFO) << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG(INFO) << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG(INFO) << "The NetState did not contain stage '" << rule.stage(i)
                << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG(INFO) << "The NetState contained a not_stage '" << rule.not_stage(i)
                << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new input or top blob to the net.  (Inputs have
// layer_id == -1, tops have layer_id >= 0.)
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param((layer_id >= 0) ?
    (new LayerParameter(param.layer(layer_id))) : NULL);
  const string& blob_name = layer_param ?
      (layer_param->top_size() > top_id ?
          layer_param->top(top_id) : "(automatic)") : param.input(top_id);
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG(INFO) << layer_param->name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
  } else {
    // Normal output.
    if (layer_param) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    } else {
      LOG(INFO) << "Input " << top_id << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    top_layer_indices_.push_back(make_pair(layer_id, blob_id));
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    if (layer_id == -1) {
      // Set the (explicitly specified) dimensions of the input blob.
      if (param.input_dim_size() > 0) {
        blob_pointer->Reshape(param.input_dim(top_id * 4),
                              param.input_dim(top_id * 4 + 1),
                              param.input_dim(top_id * 4 + 2),
                              param.input_dim(top_id * 4 + 3));
      } else {
        blob_pointer->Reshape(param.input_shape(top_id));
      }
      net_input_blob_indices_.push_back(blob_id);
      net_input_blobs_.push_back(blob_pointer.get());
    } else {
      top_id_vecs_[layer_id].push_back(blob_id);
      top_vecs_[layer_id].push_back(blob_pointer.get());
    }

  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown blob input " << blob_name
               << " (at index " << bottom_id << ") to layer " << layer_id;
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG(INFO) << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool propagate_down = true;
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0)
    propagate_down = layer_param.propagate_down(bottom_id);
  const bool need_backward = blob_need_backward_[blob_id] &&
                          propagate_down;
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG(INFO) << "Sharing parameters '" << param_name << "' owned by "
              << "layer '" << layer_names_[owner_layer_id] << "', param "
              << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Shared parameter blobs must have the same count.";
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape());
    }
    layers_[layer_id]->blobs()[param_id]->ShareData(
        *layers_[owner_layer_id]->blobs()[owner_param_id]);
  }
}

template <typename Dtype>
void Net<Dtype>::GetLearningRateAndWeightDecay() {
  LOG(INFO) << "Collecting Learning Rate and Weight Decay.";
  ParamSpec default_param_spec;
  for (int i = 0; i < layers_.size(); ++i) {
    vector<shared_ptr<Blob<Dtype> > >& layer_blobs = layers_[i]->blobs();
    for (int j = 0; j < layer_blobs.size(); ++j) {
      const ParamSpec* param_spec =
          (layers_[i]->layer_param().param_size() > j) ?
          &layers_[i]->layer_param().param(j) : &default_param_spec;
      params_lr_.push_back(param_spec->lr_mult());
      params_weight_decay_.push_back(param_spec->decay_mult());
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  if (debug_info_) {
    for (int i = 0; i < net_input_blobs_.size(); ++i) {
      InputDebugInfo(i);
    }
  }
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }

  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::ForwardPrefilled(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  // Copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return ForwardPrefilled(loss);
}

template <typename Dtype>
string Net<Dtype>::Forward(const string& input_blob_protos, Dtype* loss) {
  BlobProtoVector blob_proto_vec;
  if (net_input_blobs_.size()) {
    blob_proto_vec.ParseFromString(input_blob_protos);
    CHECK_EQ(blob_proto_vec.blobs_size(), net_input_blobs_.size())
        << "Incorrect input size.";
    for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
      net_input_blobs_[i]->FromProto(blob_proto_vec.blobs(i));
    }
  }
  ForwardPrefilled(loss);
  blob_proto_vec.Clear();
  for (int i = 0; i < net_output_blobs_.size(); ++i) {
    net_output_blobs_[i]->ToProto(blob_proto_vec.add_blobs());
  }
  string output;
  blob_proto_vec.SerializeToString(&output);
  return output;
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());

  for (int i = start; i >= end; --i) {
    if (optimize_memory_) {
      // Manually set the bottom diff to zero if it is not backpropagated.
      // If not set, they may be corrupted when memory optimization is on.
      const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[i];
      for (int j = 0; j < bottom_vec.size(); ++j)
        if (!layer_need_backward_[i] || !bottom_need_backward_[i][j]) {
          bottom_vec[j]->scale_diff(0);
        }
    }
    if (layer_need_backward_[i]) {

      //DEBUG USE
//      for (int x = 0; x < top_vecs_[i].size(); ++x){
//        LOG(INFO)<<"Layer "<<i<<" name "<<layer_names_[i]
//          <<" top blob "<<x<<" ptr: "<<top_vecs_[i][x]->gpu_diff();
//      }
//
//      for (int x = 0; x < bottom_vecs_[i].size(); ++x){
//        LOG(INFO)<<"Layer "<<i<<" name "<<layer_names_[i]
//          <<" bottom blob "<<x<<" ptr: "<<bottom_vecs_[i][x]->mutable_gpu_diff();
//      }
      //END DEBUG

      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);

      if (debug_info_) { BackwardDebugInfo(i); }

#ifdef USE_MPI
      if ((Caffe::parallel_mode() == Caffe::MPI) && (Caffe::remaining_sub_iter() == 0)) {
        for (int n = 0; n < param_layer_indices_.size(); ++n) {
          bool ready_for_sync = false;

          //decide whether we need to sync the gradient of this blob
          if ((param_layer_indices_[n].first == i)) {
            if (param_owners_[n] == -1) {
              ready_for_sync = true;
            } else {
              // this blob is a shared one, we need to make sure no more gradients will be
              // accumulated to it before transmission
              int owner_id = param_owners_[n];
              ready_for_sync = true;
              for (int m = n - 1; m >= 0; --m) {
                if ((param_owners_[m] == owner_id) && (param_layer_indices_[m].first >= end)) {
                  // there are still layers holding this shared blob,
                  // not secure the do the transmission
                  ready_for_sync = false;
                  break;
                }
              }
            }
          }
          //sync gradient
          if (ready_for_sync && layers_[i]->need_sync())
            caffe_iallreduce(
                this->params_[n]->mutable_cpu_diff(),
                this->params_[n]->count()
            );
        }
      }
#endif //USE_MPI

    }
  }
}

template <typename Dtype>
void Net<Dtype>::InputDebugInfo(const int input_id) {
  const Blob<Dtype>& blob = *net_input_blobs_[input_id];
  const string& blob_name = blob_names_[net_input_blob_indices_[input_id]];
  const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
  LOG(INFO) << "    [Forward] "
     << "Input " << blob_name << " data: " << data_abs_val_mean;
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG(INFO) << "    [Forward] "
       << "Layer " << layer_names_[layer_id] << ", top blob " << blob_name
       << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG(INFO) << "    [Forward] "
       << "Layer " << layer_names_[layer_id] << ", param blob " << blob_name
       << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG(INFO) << "    [Backward] "
        << "Layer " << layer_names_[layer_id] << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG(INFO) << "    [Backward] "
        << "Layer " << layer_names_[layer_id] << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG(INFO) << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG(INFO) << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", "
        << "param " << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      DLOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape());
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < params_.size(); ++i) {
      if (param_owners_[i] >= 0) { continue; }
      asum_data += params_[i]->asum_data();
      asum_diff += params_[i]->asum_diff();
      sumsq_data += params_[i]->sumsq_data();
      sumsq_diff += params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
        << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
        << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }

}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      DLOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_input(blob_names_[net_input_blob_indices_[i]]);
  }
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    for (int j = 0; j < bottom_id_vecs_[i].size(); ++j) {
      layer_param->add_bottom(blob_names_[bottom_id_vecs_[i][j]]);
    }
    for (int j = 0; j < top_id_vecs_[i].size(); ++j) {
      layer_param->add_top(blob_names_[top_id_vecs_[i][j]]);
    }
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::Update() {
  // First, accumulate the diffs of any shared parameters into their owner's
  // diff. (Assumes that the learning rate, weight decay, etc. have already been
  // accounted for in the current diff.)
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    if (debug_info_) { UpdateDebugInfo(i); }
    const int count = params_[i]->count();
    const Dtype* this_diff;
    Dtype* owner_diff;
    switch (Caffe::mode()) {
    case Caffe::CPU:
      this_diff = params_[i]->cpu_diff();
      owner_diff = params_[param_owners_[i]]->mutable_cpu_diff();
      caffe_add(count, this_diff, owner_diff, owner_diff);
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      this_diff = params_[i]->gpu_diff();
      owner_diff = params_[param_owners_[i]]->mutable_gpu_diff();
      caffe_gpu_add(count, this_diff, owner_diff, owner_diff);
#else
      NO_GPU;
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
  // Now, update the owned parameters.
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] >= 0) { continue; }
    if (debug_info_) { UpdateDebugInfo(i); }
    params_[i]->Update();
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

/**
 * This class is the core of memory optimization
 * It simulates an abstract ``slot'' with shared by multiple syncedmem instances.
 * The slot will be held exclusively by one syncedmem at a time.
 * This starts when the related layer writes data to this memory block and ends when the data is no-longer needed for
 * propagation.
 * During the dry-run process, a dynamic number of slots are created when deemed necessary (no empty slot is available
 * when we are acquring one).
 * By keeping track of data depedencies, we can safely make a series of blobs share the underlying storage without the
 * risk of data corruption.
 */
class SlotMeta {
public:
    SlotMeta()
      : key_(), ref_(0) { }

    SlotMeta(const string& key, int ref)
      : key_(key), ref_(ref) { }

    inline const string& key() const { return key_; }
    inline int ref() const { return ref_; }

    inline void DerefOne(){
      CHECK_GT(ref_, 0)<<"Trying to deference a free slot. Potentially this is a bug in the memory optimization process.";
      ref_ -= 1;
      if (ref_ == 0){
        key_.clear();
      }
    }

    inline void IncRef(){ref_ += 1;}

    void RefSlot(const string& key, int ref) {
      CHECK(key_.empty())<<"Referencing an non empty slot: "<<key_<<" with a new key: "<<key;
      CHECK_GT(ref, 0);
      key_ = key;
      ref_ = ref;
    }

    inline bool Empty(){
      return key_.empty();
    }

    inline bool isSlot(const string& key){return key_ == key;}

private:
    string key_;
    int ref_;
};

size_t AcquireSlot(vector<SlotMeta>& slot_vec, const string& key, int ref) {
  for (size_t i = 0 ; i < slot_vec.size(); ++i) {
    if (slot_vec[i].Empty()) {
      slot_vec[i].RefSlot(key, ref);
      return i;
    }
  }

  // no available slot, need a new one
  slot_vec.push_back(SlotMeta(key, ref));

  return slot_vec.size() - 1;
}

int FindSlot(vector<SlotMeta>& slot_vec, const string& key){
  for (int i = 0; i < slot_vec.size(); ++i){
    if (slot_vec[i].isSlot(key)){
      return i;
    }
  }
  return -1;
}

inline bool check_exclude(const std::set<string>& exclude_list, const string& blob_name){
  return exclude_list.find(blob_name) != exclude_list.end();
}


inline string create_or_link(boost::unordered_map<string, string>& record,
                             const string& blob_name, const string& suffix){
  string name = blob_name + suffix;
  if (record.find(name) != record.end()){
    return record[name];
  }else{
    record[name] = name;
    return name;
  }
}

inline void exclude_both(boost::unordered_map<string, string>& record,
                         std::set<string>& ex_set,
                         const string& blob_name){
  string root_data_name = create_or_link(record, blob_name, "_data");
  string root_diff_name = create_or_link(record, blob_name, "_diff");

  ex_set.insert(root_data_name);
  ex_set.insert(root_diff_name);
}

template <typename Dtype>
void Net<Dtype>::MemoryOptimize_v2(){
  // Pre-works
  // Check the share data/diff situation
  boost::unordered_map<string, string> share_record;
  std::set<string> excluded_names_;

  LOG(INFO)<<"Starting Memory Optimization:";

  LOG(INFO)<<"Checking data/diff sharing status";
  for (int i = 0; i < layers_.size(); ++i){
    const vector<Blob<Dtype>* >& layer_top = top_vecs_[i];
    const vector<Blob<Dtype>* >& layer_bottom = bottom_vecs_[i];

    for (int i_top = 0; i_top < layer_top.size(); ++i_top){

      const string& top_name = blob_names_[top_id_vecs_[i][i_top]];
      string root_top_data_name = create_or_link(share_record, top_name, "_data");
      string root_top_diff_name = create_or_link(share_record, top_name, "_diff");

      for (int i_bottom = 0; i_bottom < layer_bottom.size(); ++i_bottom){

        const string& bottom_name = blob_names_[bottom_id_vecs_[i][i_bottom]];
        string root_bottom_data_name = create_or_link(share_record, bottom_name, "_data");
        string root_bottom_diff_name = create_or_link(share_record, bottom_name, "_diff");

        // shared data memory blocks forms unions, we link all nodes in a union to their common root node
        if (layers_[i]->is_sharing_data(i_top, i_bottom)){
          share_record[root_top_data_name] = root_bottom_data_name;
        }

        // same goes for diff memory blocks
        if (layers_[i]->is_sharing_diff(i_top, i_bottom)){
          share_record[root_bottom_diff_name] = root_top_diff_name;
        }
      }
    }
  }

  // Color the excluded sets
  for (std::set<string>::iterator it = excluded_blob_names_.begin(); it != excluded_blob_names_.end(); ++it){
    exclude_both(share_record, excluded_names_, *it);
  }

  // Exclude input & outputs
  for (int i = 0; i < net_output_blob_indices_.size(); ++i){
    exclude_both(share_record, excluded_names_, blob_names_[net_output_blob_indices_[i]]);
    LOG(INFO)<<"excluding output "<<blob_names_[net_output_blob_indices_[i]];
  }
  for (int i = 0; i < net_input_blob_indices_.size(); ++i){
    exclude_both(share_record, excluded_names_, blob_names_[net_input_blob_indices_[i]]);
    LOG(INFO)<<"excluding input "<<blob_names_[net_input_blob_indices_[i]];
  }

  // Exclude network sources, aka data layers
  for (int i = 0; i < layers_.size(); ++i) {
    if (bottom_vecs_[i].size() == 0){
      for (int i_top = 0; i_top < top_vecs_[i].size(); ++i_top){
        const string& top_name = blob_names_[top_id_vecs_[i][i_top]];
        exclude_both(share_record, excluded_names_, top_name);
        LOG(INFO)<<"excluding data layer output "<<top_name;
      }
    }
  }

  // Exclude all losses
  for (int i = 0; i < layers_.size(); ++i){
    for (int i_top = 0; i_top < top_vecs_[i].size(); ++i_top){
      if (layers_[i]->loss(i_top)){
        const string& top_name = blob_names_[top_id_vecs_[i][i_top]];
        exclude_both(share_record, excluded_names_, top_name);
        LOG(INFO)<<"excluding loss "<<top_name;
      }
    }
  }

  // Pre-works done
  // Dry run to determine dependencies.

  vector<SlotMeta> slots;
  boost::unordered_map<string, int> slot_index;

  int direction = 1;
  string str_direction = "forward";
  for (int i = 0; i >= 0;){
    const vector<Blob<Dtype>* >& layer_output = (direction>0)?top_vecs_[i]:bottom_vecs_[i];
    const vector<Blob<Dtype>* >& layer_input = (direction>0)?bottom_vecs_[i]:top_vecs_[i];

    LOG(INFO)<< "layer " <<i<< " layer name: "<<layer_names_[i]<< " direction: "<<str_direction;
    string suffix = (direction>0)?"_data":"_diff";

    // Find slot for each layer output data
    for (int i_out = 0; i_out < layer_output.size(); ++i_out){
      const string& output_name = blob_names_[(direction>0)?top_id_vecs_[i][i_out]:bottom_id_vecs_[i][i_out]];

      string root_full_name = create_or_link(share_record, output_name, suffix);

      if (check_exclude(excluded_names_, root_full_name)) continue;

      string output_full_name = output_name + suffix;

      // not excluded, let's do the math
      int idx = FindSlot(slots, output_full_name);
      if (idx == -1){
        if (root_full_name == output_full_name){
          // not sharing data
          idx = (int)AcquireSlot(slots, output_full_name, 1);
          slot_index[output_full_name] = idx;
          LOG(INFO)<<"blob "<<output_full_name<<" acquired new slot "<<idx;
        }else{
          // sharing data with its root
          slot_index[output_full_name] = slot_index[root_full_name];
          slots[slot_index[root_full_name]].IncRef();
          LOG(INFO)<<"blob "<<output_full_name<<" shares its root " <<root_full_name<<"'s slot: "<<slot_index[output_full_name];
        }
      } else {
        // in-place operations
        slots[idx].IncRef();
      }

    }

    // Deref the layer's output if necessary
    for (int i_in = 0; i_in < layer_input.size(); ++i_in){
      const string& input_name = blob_names_[(direction>0)?bottom_id_vecs_[i][i_in]:top_id_vecs_[i][i_in]];
      string root_full_name = create_or_link(share_record, input_name, suffix);

      if (check_exclude(excluded_names_, root_full_name)) continue;

      string input_full_name = input_name + suffix;

      if (phase_ == TRAIN && layer_need_backward_[i] && direction > 0) {
        LOG(INFO)<<"skipping deref";
        continue;
      }

      int idx = FindSlot(slots, root_full_name);
      slots[idx].DerefOne();
      LOG(INFO)<<"deref slot "<<idx<<" held by blob "<<root_full_name;
    }

    // reverse once we reach the end of forward
    if (direction > 0 && i == layers_.size() - 1) {
      direction = -1;
      str_direction = "backward";
    }else{
      i += direction;
    }

  }


  // Memory assignment
  shared_storage_.resize(slots.size());
  for (int i_mem = 0; i_mem < shared_storage_.size(); i_mem++){
    shared_storage_[i_mem].reset(new SyncedMemory(1));
  }

  size_t count_raw = 0;
  size_t count_opt = 0;
  for (int i_blob = 0; i_blob < blobs_.size(); ++i_blob){
    const string& name = blob_names_[i_blob];
    const size_t bytes = blobs_[i_blob]->count() * sizeof(Dtype);
    count_raw += bytes * 2;
    int idx = -1;

    // all blobs in the same slot share a same externally hosted SyncedMem instance
    // we will keep track of the estimated memory usage reduction while linking them to the SyncedMem
    if (slot_index.find(name + "_data") != slot_index.end()) {
      idx = slot_index[name + "_data"];
      blobs_[i_blob]->SetDataStorage(shared_storage_[idx]);
      shared_storage_[idx]->Resize(bytes);
    } else {
      count_opt += bytes;
    }
    LOG(INFO) << "blob " << i_blob
        << " name " << blob_names_[i_blob]
        << " data idx " << idx;
    if (slot_index.find(name + "_diff") != slot_index.end()) {
      idx = slot_index[name + "_diff"];
      blobs_[i_blob]->SetDiffStorage(shared_storage_[idx]);
      shared_storage_[idx]->Resize(bytes);
    } else {
      count_opt += bytes;
    }
    LOG(INFO) << "blob " << i_blob
        << " name " << blob_names_[i_blob]
        << " diff idx " << idx;
  }

  for (int i_mem = 0; i_mem < shared_storage_.size(); i_mem++){
    LOG(INFO) << "storage memory slot " << i_mem
        << " size " << shared_storage_[i_mem]->size();
    count_opt += shared_storage_[i_mem]->size();
  }

  LOG(INFO) << "raw memory " << count_raw << " opt memory " << count_opt;

}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
