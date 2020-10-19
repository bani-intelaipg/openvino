
#include <vector>
#include <memory>
#include <string>
#include <samples/common.hpp>

#include <inference_engine.hpp>

#include <ie_core.hpp>


#include "ngraph/ngraph.hpp"

#include <ngraph/opsets/opset.hpp>
#include <ngraph/pass/manager.hpp>


#include "ngraph/function.hpp"

#include "ngraph/file_util.hpp"

using namespace std;
using namespace ngraph;

std::shared_ptr<Function> createNgraphFunction2() {

    auto paramNode1 = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, 1, 512, 256}}));
    paramNode1->set_friendly_name("Parameter1");
    auto paramNode2 = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, 1, 256, 512}}));
    paramNode2->set_friendly_name("Parameter2");

    // std::shared_ptr<Node> mulconst1Node = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{0.176777});
    // mulconst1Node->set_friendly_name("Mul1Const1");
    // std::shared_ptr<Node> mulconst2Node = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{0.176777});
    // mulconst2Node->set_friendly_name("Mul1Const2");

    // std::shared_ptr<Node> mul1Node = std::make_shared<op::v1::Multiply>(paramNode1->output(0), mulconst1Node->output(0));
    // mul1Node->set_friendly_name("Mul1");
    std::shared_ptr<Node> add1ConstantNode = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{-0.0883883});
    add1ConstantNode->set_friendly_name("Add1Const");
    std::shared_ptr<Node> add1Node = std::make_shared<op::v1::Add>(add1ConstantNode->output(0), paramNode1->output(0));

    // std::shared_ptr<Node> mul2Node = std::make_shared<op::v1::Multiply>(paramNode2->output(0), mulconst2Node->output(0));
    // mul2Node->set_friendly_name("Mul2");
    std::shared_ptr<Node> add2ConstantNode = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{-0.0883883});
    add2ConstantNode->set_friendly_name("Add2Const");
    std::shared_ptr<Node> add2Node = std::make_shared<op::v1::Add>(add2ConstantNode->output(0), paramNode2->output(0));

    // -------ngraph function--
    auto result1 = std::make_shared<op::Result>(add1Node->output(0));
    auto result2 = std::make_shared<op::Result>(add2Node->output(0));
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{result1, result2}, ngraph::ParameterVector{ paramNode1, paramNode2 }, "Func1");

    return fnPtr;

}

std::shared_ptr<Function> createNgraphFunction1() {

    auto paramNode1 = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, 1, 512, 256}}));
    paramNode1->set_friendly_name("Parameter1");
    auto paramNode2 = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, 1, 256, 512}}));
    paramNode2->set_friendly_name("Parameter2");

    std::shared_ptr<Node> mulConstantNode = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{0.176777});
    mulConstantNode->set_friendly_name("CommonMulConst");

    std::shared_ptr<Node> mul1Node = std::make_shared<op::v1::Multiply>(paramNode1->output(0), mulConstantNode->output(0));
    mul1Node->set_friendly_name("Mul1");
    std::shared_ptr<Node> add1ConstantNode = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{-0.0883883});
    add1ConstantNode->set_friendly_name("Add1Const");
    std::shared_ptr<Node> add1Node = std::make_shared<op::v1::Add>(add1ConstantNode->output(0), mul1Node->output(0));

    std::shared_ptr<Node> mul2Node = std::make_shared<op::v1::Multiply>(paramNode2->output(0), mulConstantNode->output(0));
    mul2Node->set_friendly_name("Mul2");
    std::shared_ptr<Node> add2ConstantNode = std::make_shared<op::Constant>(element::Type_t::f32, Shape(std::vector<size_t>{{1,1,1,1}}), std::vector<float>{-0.0883883});
    add2ConstantNode->set_friendly_name("Add2Const");
    std::shared_ptr<Node> add2Node = std::make_shared<op::v1::Add>(add2ConstantNode->output(0), mul2Node->output(0));

    // -------ngraph function--
    auto result1 = std::make_shared<op::Result>(add1Node->output(0));
    auto result2 = std::make_shared<op::Result>(add2Node->output(0));
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{result1, result2}, ngraph::ParameterVector{ paramNode1, paramNode2 }, "Func1");

    return fnPtr;

}

using namespace InferenceEngine;

int main(int argc, char *argv[]) {

    try {

        const std::string device_name = "CPU";
        shared_ptr<ngraph::Function> func; // the func to pass to CNNNetwork ctor

        func = createNgraphFunction1();

        std::cout << "BEFORE InferenceEngine::CNNNetwork(func), func=" << func->get_friendly_name() << ", output_size=" << func->get_output_size() << " ==>>\n";
        for (auto aNodeShPtr : func->get_ordered_ops()) { std::cout << aNodeShPtr->get_friendly_name() << " (" << aNodeShPtr->get_name() << " / " << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n";

        CNNNetwork m_network = InferenceEngine::CNNNetwork(func);

        std::cout << "\nAFTER InferenceEngine::CNNNetwork(func), CNN network details ...\n";
        std::cout << "CNN layer count = " << m_network.layerCount() << "\n";
        // CNNNetworkNGraphImpl m_network2 = (CNNNetworkNGraphImpl)m_network;
        // auto layers = m_network2.allLayers();
        // std::cout << "CNN Layers => "; for (auto const& pair : layers) { std::cout << pair.first << "->" << pair.second << ", "; } std::cout << "\n";

        //m_network.serialize("temp_cnn.xml");

        auto outputs = m_network.getOutputsInfo(); // OutputsDataMap
        std::cout << "CNN Outputs => "; for (auto const& pair : outputs) { std::cout << pair.first << "->" << pair.second << ", "; } std::cout << "\n";
        auto inputs = m_network.getInputsInfo(); // OutputsDataMap
        std::cout << "CNN Inputs => "; for (auto const& pair : inputs) { std::cout << pair.first << "->" << pair.second << ", "; } std::cout << "\n";

        auto const& cnn_ngfunc = m_network.getFunction(); // shared_ptr<ngraph::Function>
        #if 1
        // Bani
        auto& results = cnn_ngfunc->get_results();
        auto& params = cnn_ngfunc->get_parameters();
        std::cout << "\ncnn_ngfunc = " << cnn_ngfunc->get_friendly_name() << " (" << cnn_ngfunc->get_name() << ") " <<
            ", output_size=" << results.size() << 
            ", input_size=" << params.size() <<
            " ==>>\n";
        for (auto aNodeShPtr : results) { std::cout << aNodeShPtr->get_friendly_name() << " (" << aNodeShPtr->get_name() << " / " << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n";
        for (auto aNodeShPtr : params) { std::cout << aNodeShPtr->get_friendly_name() << " (" << aNodeShPtr->get_name() << " / " << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n";
        #endif
        
        InferenceEngine::Core ie; // Load inference engine instance
        ExecutableNetwork exe_network = ie.LoadNetwork(m_network, device_name); //  Load model to the plugin (BACKEND_NAME)

        std::cout << "\nAFTER LoadNetwork, " << cnn_ngfunc->get_friendly_name() << " ==>\n";
        //for (auto aNodeShPtr : m_network.getFunction()->get_ordered_ops()) { std::cout << aNodeShPtr->get_name() << " (" << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n\n";
        std::cout << "    outs = "; for (auto const& pair : exe_network.GetOutputsInfo()) { std::cout << pair.first << ", "; } std::cout << "\n";
        std::cout << "    ins = "; for (auto const& pair : exe_network.GetInputsInfo()) { std::cout << pair.first << ", "; } std::cout << "\n";

        //InferRequest infer_request = executable_network.CreateInferRequest();

        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\nProgram Completed." << std::endl;
    return EXIT_SUCCESS;

}

