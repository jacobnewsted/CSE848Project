using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace GANLibrary
{
    class Generator
    {
        int TargetWidth = 0;
        int TargetHeight = 0;

        Dictionary<string, TFOutput> Outputs = new Dictionary<string, TFOutput>();

        private Generator() { }
        public Generator(int targetWidth, int targetHeight, TFGraph TFGraph)
        {
            TargetWidth = targetWidth;
            TargetHeight = targetHeight;

            TFOutput Z = TFGraph.PlaceholderV2(TFDataType.Float, new TFShape(new long[] { 100 }), "Z");

            TFOutput G_W1 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 100, 128 })));
            TFOutput G_b1 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { 128 })));

            TFOutput G_W2 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 128, TargetWidth * TargetHeight })));
            TFOutput G_b2 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { TargetWidth * TargetHeight })));

            Outputs.Add("Z", Z);
            Outputs.Add("G_W1", G_W1);
            Outputs.Add("G_W2", G_W2);
            Outputs.Add("G_b1", G_b1);
            Outputs.Add("G_b2", G_b2);
        }

        public TFOutput GenerateProb(TFGraph TFGraph, TFOutput z)
        {
            
            TFOutput G_h1 = TFGraph.Relu(TFGraph.Add(TFGraph.MatMul(z, Outputs["G_W1"]), Outputs["G_b1"]));
            TFOutput G_log_prob = TFGraph.Add(TFGraph.MatMul(G_h1, Outputs["G_W2"]), Outputs["G_b2"]);
            TFOutput G_prob = TFGraph.Sigmoid(G_log_prob);
            return G_prob;
        }
    }
}
