// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the CNNNetworkIterator class
 * 
 * @file ie_cnn_network_iterator.hpp
 */
#pragma once
#include <iterator>
#include <list>
#include <unordered_set>
#include <utility>

#include "ie_api.h"
#include "cpp/ie_cnn_network.h"
#include "ie_locked_memory.hpp"

#include <legacy/ie_layers.h>
#include <legacy/cnn_network_impl.hpp>

namespace InferenceEngine {
namespace details {

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class enables range loops for CNNNetwork objects
 */
class INFERENCE_ENGINE_INTERNAL("Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1")
CNNNetworkIterator {
    IE_SUPPRESS_DEPRECATED_START

    std::unordered_set<CNNLayer*> visited;
    std::list<CNNLayerPtr> nextLayersTovisit;
    InferenceEngine::CNNLayerPtr currentLayer;
    const ICNNNetwork* network = nullptr;

    void init(const ICNNNetwork* network) {
        if (network == nullptr) THROW_IE_EXCEPTION << "ICNNNetwork object is nullptr";
        // IE_ASSERT(dynamic_cast<const details::CNNNetworkImpl*>(network) != nullptr);
        this->network = network; // Bani
        InputsDataMap inputs;
        network->getInputsInfo(inputs);
        // Bani: scan for at least one input which does have a NEXT
        for(auto& ip : inputs) {
            auto& nextLayers = getInputTo(ip.second->getInputData());
            if (!nextLayers.empty()) {
                currentLayer = nextLayers.begin()->second;
                nextLayersTovisit.push_back(currentLayer);
                visited.insert(currentLayer.get());
                break;
            }
        }
    }

public:
    /**
     * iterator trait definitions
     */
    typedef std::forward_iterator_tag iterator_category;
    typedef CNNLayerPtr value_type;
    typedef int difference_type;
    typedef CNNLayerPtr pointer;
    typedef CNNLayerPtr reference;

    /**
     * @brief Default constructor
     */
    CNNNetworkIterator() = default;
    /**
     * @brief Constructor. Creates an iterator for specified CNNNetwork instance.
     * @param network Network to iterate. Make sure the network object is not destroyed before iterator goes out of
     * scope.
     */
    explicit CNNNetworkIterator(const ICNNNetwork* network) {
        init(network);
    }

    explicit CNNNetworkIterator(const CNNNetwork & network) {
        const auto & inetwork = static_cast<const InferenceEngine::ICNNNetwork&>(network);
        init(&inetwork);
    }

    const CNNLayerPtr& getCurrentLayer() {
        return currentLayer;
    }

    /**
     * @brief Performs pre-increment
     * @return This CNNNetworkIterator instance
     */
    CNNNetworkIterator& operator++() {
        currentLayer = next();
        return *this;
    }

    /**
     * @brief Performs post-increment.
     * Implementation does not follow the std interface since only move semantics is used
     */
    void operator++(int) {
        currentLayer = next();
    }

    /**
     * @brief Checks if the given iterator is not equal to this one
     * @param that Iterator to compare with
     * @return true if the given iterator is not equal to this one, false - otherwise
     */
    bool operator!=(const CNNNetworkIterator& that) const {
        return !operator==(that);
    }

    /**
     * @brief Gets const layer pointer referenced by this iterator
     */
    const CNNLayerPtr& operator*() const {
        if (nullptr == currentLayer) {
            THROW_IE_EXCEPTION << "iterator out of bound";
        }
        return currentLayer;
    }

    /**
     * @brief Gets a layer pointer referenced by this iterator
     */
    CNNLayerPtr& operator*() {
        if (nullptr == currentLayer) {
            THROW_IE_EXCEPTION << "iterator out of bound";
        }
        return currentLayer;
    }
    /**
     * @brief Compares the given iterator with this one
     * @param that Iterator to compare with
     * @return true if the given iterator is equal to this one, false - otherwise
     */
    bool operator==(const CNNNetworkIterator& that) const {
        //return network == that.network && currentLayer == that.currentLayer;
        bool retVal;
        if(currentLayer == nullptr && that.currentLayer == nullptr) {
            retVal = true;
        } else {
            retVal = network == that.network && currentLayer == that.currentLayer;
        }
        //BANI_DBG: std::cout << "     operator== called, retVal=" << retVal << ", this=" << this << ", this.currentLayer = " << ((currentLayer)?(currentLayer->name):("NULL")) << 
        //BANI_DBG:     ", that=" << &that << ", that.currentLayer = " << ((that.currentLayer)?(that.currentLayer->name):("NULL")) <<"\n";
        return retVal;
    }

private:
    /**
     * @brief implementation based on BFS
     */
    CNNLayerPtr next() {
        if (nextLayersTovisit.empty()) {
            return nullptr;
        }

        auto nextLayer = nextLayersTovisit.front();
        nextLayersTovisit.pop_front();

        // visit child that not visited
        for (auto&& output : nextLayer->outData) {
            for (auto&& child : getInputTo(output)) {
                if (visited.find(child.second.get()) == visited.end()) {
                    nextLayersTovisit.push_back(child.second);
                    visited.insert(child.second.get());
                }
            }
        }

        // visit parents
        for (auto&& parent : nextLayer->insData) {
            auto parentLayer = getCreatorLayer(parent.lock()).lock();
            if (parentLayer && visited.find(parentLayer.get()) == visited.end()) {
                nextLayersTovisit.push_back(parentLayer);
                visited.insert(parentLayer.get());
            }
        }

        // Bani
        // check possibly for other disjoint univisited input subtree
        if(nextLayersTovisit.empty()) {
            //BANI_DBG: std::cout << "     CNNNetworkIterator(ICNNNetwork*) next(), checking more input subtree\n";
            InputsDataMap inputs;
            network->getInputsInfo(inputs);
            if (!inputs.empty()) {
                for(auto itr=inputs.begin(); itr!=inputs.end(); ++itr) { // Bani
                    //BANI_DBG: std::cout << "    trying input = " << itr->first << "\n";
                    auto& nextLayers = getInputTo(itr->second->getInputData());
                    if (nextLayers.empty()) {
                        continue;
                    }
                    currentLayer = nextLayers.begin()->second;
                    if (visited.find(currentLayer.get()) == visited.end()) {
                        //BANI_DBG: std::cout << "    unvisited, exploring input = " << itr->first << "\n";
                        //BANI_DBG: std::cout << "    currentLayer = " << currentLayer->name <<"\n";
                        nextLayersTovisit.push_back(currentLayer);
                        visited.insert(currentLayer.get());
                        break;
                    }
                }
            }
        }

        return nextLayersTovisit.empty() ? nullptr : nextLayersTovisit.front();
    }

    IE_SUPPRESS_DEPRECATED_END
};
}  // namespace details
}  // namespace InferenceEngine
