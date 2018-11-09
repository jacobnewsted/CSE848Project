using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace GANLibrary
{
    class Discriminator
    {
        int TargetWidth = 0;
        int TargetHeight = 0;

        List<string[]> realImages = new List<string[]>();

        Dictionary<string, TFOutput> Outputs = new Dictionary<string, TFOutput>();
        private Discriminator() { }

        public Discriminator(int targetWidth, int targetHeight, TFGraph TFGraph)
        {
            TargetWidth = targetWidth;
            TargetHeight = targetHeight;

            TFOutput X = TFGraph.PlaceholderV2(TFDataType.Float, new TFShape(new long[] { TargetWidth * TargetHeight }), "X");

            TFOutput D_W1 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 784, 128 })));
            TFOutput D_b1 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { 128 })));

            TFOutput D_W2 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 128, 1 })));
            TFOutput D_b2 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { 128 })));

            Outputs.Add("X", X);
            Outputs.Add("D_W1", D_W1);
            Outputs.Add("D_W2", D_W2);
            Outputs.Add("D_b1", D_b1);
            Outputs.Add("D_b2", D_b2);
        }

        public TFOutput[] DiscrimProb(TFGraph TFGraph, TFOutput x)
        {
            TFOutput[] tFOutputs = new TFOutput[2];
            TFOutput D_h1 = TFGraph.Relu(TFGraph.Add(TFGraph.MatMul(x, Outputs["D_W1"]), Outputs["D_b1"]));
            tFOutputs[1] = TFGraph.Add(TFGraph.MatMul(Outputs["D_h1"], Outputs["D_W2"]), Outputs["D_b2"]);
            tFOutputs[0] = TFGraph.Sigmoid(tFOutputs[1]);

            return tFOutputs;
        }
    }
}
