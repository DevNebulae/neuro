using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro;

namespace Neuro.Tests
{
    [TestClass]
    public class XORNetTest
    {
        private NeuralNetwork net;

        [TestInitialize]
        public void Initialize()
        {
            this.net = new NeuralNetwork(new int[] { 2, 4, 1 });
        }

        [TestMethod]
        public void CreateXORTest()
        {
            Assert.IsNotNull(net);
        }

        private void Train(NeuralNetwork net, double[] inputs, double[] targets)
        {
            net.FeedForward(inputs);
            net.BackPropagate(targets);
        }

        private double[] Test(NeuralNetwork net, double[] inputs)
        {
            net.FeedForward(inputs);
            return net.Results();
        }

        [TestMethod]
        public void TrainTest()
        {
            for (int i = 0; i < 1000; ++i)
            {
                Train(this.net, new double[] { 0.0, 0.0 }, new double[] { 0.0 });
                Train(this.net, new double[] { 1.0, 0.0 }, new double[] { 1.0 });
                Train(this.net, new double[] { 0.0, 1.0 }, new double[] { 1.0 });
                Train(this.net, new double[] { 1.0, 1.0 }, new double[] { 0.0 });
            }

            Assert.AreEqual(Test(this.net, new double[] { 0.0, 0.0 })[0], 0.0, 0.07);
            Assert.AreEqual(Test(this.net, new double[] { 0.0, 1.0 })[0], 1.0, 0.07);
            Assert.AreEqual(Test(this.net, new double[] { 1.0, 0.0 })[0], 1.0, 0.07);
            Assert.AreEqual(Test(this.net, new double[] { 1.0, 1.0 })[0], 0.0, 0.07);
        }
    }
}
