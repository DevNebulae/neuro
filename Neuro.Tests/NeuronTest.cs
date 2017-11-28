using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro;

namespace Neuro.Tests
{
    [TestClass]
    public class NeuronTest
    {
        private Neuron neuron;

        [TestInitialize]
        public void NeuronTestInitialize()
        {
            neuron = new Neuron(null, null, 0, new Random());
        }

        [TestMethod]
        public void CreateTest()
        {
            Assert.IsNotNull(neuron);
        }
    }
}
