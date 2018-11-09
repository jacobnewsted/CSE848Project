using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorFlow;

namespace GANLibrary
{
    public class GAN
    {
        //Generator gen = null;
        //Discriminator dis = null;

        TFSession TFSession { get; set; }
        TFGraph TFGraph { get; set; }
        int TargetWidth { get; set; }
        int TargetHeight { get; set; }

        Dictionary<string, TFOutput> Generator = new Dictionary<string, TFOutput>();
        Dictionary<string, TFOutput> Discriminator = new Dictionary<string, TFOutput>();

        /*public GAN(string ganToLoad = "")
        {
            TFSession = new TFSession();
            TFGraph = TFSession.Graph;
            gen = new Generator(10, 10, TFGraph);
            dis = new Discriminator(10, 10, TFGraph);
        }*/

        public GAN(string ganToLoad = "")
        {
            if (ganToLoad == "")
            {
                TFSession = new TFSession();
                TFGraph = TFSession.Graph;
            }
            else
            {
                TFGraph = new TFGraph();
                TFGraph.Import(File.ReadAllBytes(ganToLoad));
                TFSession = new TFSession(TFGraph);
                //TFSession.GetRunner().AddInput(TFGraph["input"][0], tensor);
                //TFSession.GetRunner().Fetch(TFGraph["output"][0]);
            }

            TargetWidth = 10;
            TargetHeight = 10;

            // Initialize discriminator
            TFOutput X = TFGraph.PlaceholderV2(TFDataType.Float, new TFShape(new long[] { 784 }));
            TFOutput D_W1 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 784, 128 })));
            TFOutput D_b1 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { 128 })));

            TFOutput D_W2 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 128, 1 })));
            TFOutput D_b2 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { 128 })));

            Generator.Add("X", X);
            Generator.Add("D_W1", D_W1);
            Generator.Add("D_W2", D_W2);
            Generator.Add("D_b1", D_b1);
            Generator.Add("D_b2", D_b2);


            TFOutput Z = TFGraph.PlaceholderV2(TFDataType.Float, new TFShape(new long[] { 100 }), "Z");
            TFOutput G_W1 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 100, 128 })));
            TFOutput G_b1 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { 128 })));

            TFOutput G_W2 = TFGraph.Variable(TFGraph.RandomUniform(new TFShape(new long[2] { 128, TargetWidth * TargetHeight })));
            TFOutput G_b2 = TFGraph.Variable(TFGraph.Zeros(new TFShape(new long[1] { TargetWidth * TargetHeight })));

            Discriminator.Add("Z", Z);
            Discriminator.Add("G_W1", G_W1);
            Discriminator.Add("G_W2", G_W2);
            Discriminator.Add("G_b1", G_b1);
            Discriminator.Add("G_b2", G_b2);
        }

        public TFOutput GeneratorOutput(TFOutput z)
        {
            TFOutput toReturn;
            TFOutput G_h1 = TFGraph.Relu(TFGraph.Add(TFGraph.MatMul(z, Generator["G_W1"]), Generator["G_b1"]));
            TFOutput G_log_prob = TFGraph.Add(TFGraph.MatMul(G_h1, Generator["G_W2"]), Generator["G_b2"]);
            toReturn = TFGraph.Sigmoid(G_log_prob);
            return toReturn;
        }

        public TFOutput[] DiscriminatorOutput(TFOutput x)
        {
            TFOutput[] toReturn = new TFOutput[2];
            TFOutput D_h1 = TFGraph.Relu(TFGraph.Add(TFGraph.MatMul(x, Discriminator["D_W1"]), Discriminator["D_b1"]));
            toReturn[1] = TFGraph.Add(TFGraph.MatMul(D_h1, Discriminator["D_W2"]), Discriminator["D_b2"]);
            toReturn[0] = TFGraph.Sigmoid(toReturn[1]);

            return toReturn;
        }

        public void InitializeTensors()
        {
            Generator.Add("G_Sample", GeneratorOutput(Generator["Z"]));
            TFOutput[] real = DiscriminatorOutput(Discriminator["X"]);
            TFOutput[] fake = DiscriminatorOutput(Generator["G_Sample"]);

            Discriminator.Add("D_Real", real[0]);
            Discriminator.Add("D_Logit_Real", real[1]);
            Discriminator.Add("D_Fake", fake[0]);
            Discriminator.Add("D_Logit_Fake", fake[1]);

            TFOutput D_Loss = TFGraph.ReduceMean(TFGraph.Add(TFGraph.Log(real[0]), TFGraph.Log(TFGraph.Sub(TFGraph.Const(1.0), fake[0]))));
            TFOutput G_Loss = TFGraph.ReduceMean(TFGraph.Log(fake[0]));
        }

        public void Train(int generations)
        {
        }
    }
}
