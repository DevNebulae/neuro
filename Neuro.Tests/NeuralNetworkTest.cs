using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro;

namespace Neuro.Tests
{
    [TestClass]
    public class NeuralNetworkTest
    {
        private NeuralNetwork network;

        [TestInitialize]
        public void NetTestInitialize()
        {
            network = new NeuralNetwork(new int[] { 2, 1 });
        }

        [TestMethod]
        public void CreateNetTest()
        {
            Assert.IsNotNull(network);
        }

        [TestMethod]
        public void NeuronCountTest()
        {
            Neuron[] layer;
            layer = network.GetLayer(0);
            Assert.IsNotNull(layer);
            Assert.AreEqual(3, layer.Length);

            layer = network.GetLayer(1);
            Assert.IsNotNull(layer);
            Assert.AreEqual(2, layer.Length);
        }

        [TestMethod]
        public void ResultsTest()
        {
            double[] res = network.Results();
            Assert.AreEqual(1, res.Length);
        }

        [TestMethod]
        public void BiasNeuronTest()
        {
            Neuron[] layer;
            Neuron bias;

            layer = network.GetLayer(0);
            bias = layer[2];
            Assert.IsNotNull(bias);
            Assert.AreEqual(1.0, bias.Output, 0.0001);

            layer = network.GetLayer(1);
            bias = layer[1];
            Assert.IsNotNull(bias);
            Assert.AreEqual(1.0, bias.Output, 0.0001);
        }

    }
}
