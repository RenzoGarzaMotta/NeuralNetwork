use serde::{Serialize, Deserialize};
use plotly::common::{Mode, Title};
use plotly::{Plot, Scatter};
use csv::ReaderBuilder;
use std::any::Any;
use std::fs::File;
use std::time::Instant;
use glob::glob;
use rand::Rng;

#[allow(dead_code)]
enum NeuralNetworkMode {
    Train,
    Test,
}

#[derive(Serialize, Deserialize, Clone)]
enum Activation {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Elu(f64),  // requires alpha
    Softplus,
    Softsign,
    Gaussian,
    // Softmax,
    Swish,
}

fn activate(x: f64, act: &Activation) -> f64 {
    match act {
        Activation::Identity => x,
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Tanh => x.tanh(),
        Activation::Relu => if x > 0.0 { x } else { 0.0 },
        Activation::LeakyRelu => if x > 0.0 { x } else { 0.01 * x },
        Activation::Elu(alpha) => if x > 0.0 { x } else { alpha * ((x).exp() - 1.0) },
        Activation::Softplus => (1.0 + x.exp()).ln(),
        Activation::Softsign => x / (1.0 + x.abs()),
        Activation::Gaussian => (-x.powi(2)).exp(),
        Activation::Swish => x * (1.0 / (1.0 + (-x).exp())),
        // Activation::Softmax => {
        //     let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        //     let exps: Vec<f64> = xs.iter().map(|&x| (x - max_x).exp()).collect();
        //     let sum: f64 = exps.iter().sum();
        //     exps.iter().map(|&e| e / sum).collect()
        // },
    }
}

fn activate_derivative(x: f64, act: &Activation) -> f64 {
    match act {
        Activation::Identity => 1.0,
        Activation::Sigmoid => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
        Activation::Tanh => 1.0 - x.tanh().powi(2),
        Activation::Relu => if x > 0.0 { 1.0 } else { 0.0 },
        Activation::LeakyRelu => if x > 0.0 { 1.0 } else { 0.01 },
        Activation::Elu(alpha) => if x > 0.0 { 1.0 } else { activate(x, act) + alpha },
        Activation::Softplus => 1.0 / (1.0 + (-x).exp()), // = sigmoid(x)
        Activation::Softsign => 1.0 / (1.0 + x.abs()).powi(2),
        Activation::Gaussian => -2.0 * x * (-x.powi(2)).exp(),
        Activation::Swish => {
            let s = 1.0 / (1.0 + (-x).exp());
            s + x * s * (1.0 - s)
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
enum Loss {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Hinge,
    Huber(f64),   // delta parameter
    KLDivergence,
}

#[allow(dead_code)]
fn loss(y_true: &[f64], y_pred: &[f64], loss_fn: &Loss) -> f64 {
    match loss_fn {
        Loss::MeanSquaredError => {
            y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>() / y_true.len() as f64
        }
        Loss::MeanAbsoluteError => {
            y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| (t - p).abs())
                .sum::<f64>() / y_true.len() as f64
        }
        Loss::BinaryCrossEntropy => {
            y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| -t * p.ln() - (1.0 - t) * (1.0 - p).ln())
                .sum::<f64>() / y_true.len() as f64
        }
        Loss::CategoricalCrossEntropy => {
            // assumes y_true is one-hot
            -y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| if *t > 0.0 { t * p.ln() } else { 0.0 })
                .sum::<f64>()
        }
        Loss::Hinge => {
            y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| (1.0 - t * p).max(0.0))
                .sum::<f64>() / y_true.len() as f64
        }
        Loss::Huber(delta) => {
            y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| {
                    let err = t - p;
                    if err.abs() <= *delta {
                        0.5 * err.powi(2)
                    } else {
                        delta * (err.abs() - 0.5 * delta)
                    }
                })
                .sum::<f64>() / y_true.len() as f64
        }
        Loss::KLDivergence => {
            y_true.iter()
                .zip(y_pred)
                .map(|(t, p)| if *t > 0.0 { t * (t / p).ln() } else { 0.0 })
                .sum::<f64>()
        }
    }
}

fn loss_derivative(y_true: &[f64], y_pred: &[f64], loss_fn: &Loss) -> Vec<f64> {
    match loss_fn {
        Loss::MeanSquaredError => {
            y_true.iter().zip(y_pred)
                .map(|(t, p)| 2.0 * (p - t) / y_true.len() as f64)
                .collect()
        }
        Loss::MeanAbsoluteError => {
            y_true.iter().zip(y_pred)
                .map(|(t, p)| if p > t { 1.0 } else { -1.0 })
                .collect()
        }
        Loss::BinaryCrossEntropy => {
            y_true.iter().zip(y_pred)
                .map(|(t, p)| (p - t) / (p * (1.0 - p) * y_true.len() as f64))
                .collect()
        }
        Loss::CategoricalCrossEntropy => {
            // derivative assumes softmax output
            y_pred.iter().zip(y_true)
                .map(|(p, t)| (p - t) / y_true.len() as f64)
                .collect()
        }
        Loss::Hinge => {
            y_true.iter().zip(y_pred)
                .map(|(t, p)| if 1.0 - t * p > 0.0 { -t } else { 0.0 })
                .collect()
        }
        Loss::Huber(delta) => {
            y_true.iter().zip(y_pred)
                .map(|(t, p)| {
                    let err = p - t;
                    if err.abs() <= *delta { err } else { delta * err.signum() }
                })
                .collect()
        }
        Loss::KLDivergence => {
            y_true.iter().zip(y_pred)
                .map(|(t, p)| if *t > 0.0 {t / p} else { 0.0 })
                .collect()
        }
    }
}

struct Neuron{
    weights: Vec<f64>,
    bias: f64,
    z: f64,            // pre-activation
    output: f64,       // post-activation
    activation: Activation, //Activation Function Chosen
}

trait LayerTrait {
    fn forward(&mut self, input: Vec<f64>) -> Vec<f64>;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
}

// Structure to represent the input layer only.
struct InputLayer{
    inputs: Vec<f64>,
}

impl LayerTrait for InputLayer{
    //Pass input data forward with no modifications
    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.inputs = input.clone();
        input
    }
    
    //Support for downcasting Box<dyn LayerTrait> to InputLayer
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

}

// Structure to represent any hidden layers or output layer.
struct Layer{
    neurons:Vec<Neuron>,
}

// Structure to represent the chosen configuration for a given hidden/output layer
// size: number of neurons desired for the layer.
// activation function to be used for the layer.
struct LayerConfig {
    size: usize,
    activation: Activation,
}

impl LayerTrait for Layer{
    //Calculate the output of each neuron during forward propagation
    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut outputs = Vec::new();
        for neuron in &mut self.neurons { // For a given neuron in a layer...
            let z: f64 = neuron
                .weights
                .iter()// Iterate through the weights of all the input neurons.
                .zip(&input) // Pair each weight to a value of an input neuron.
                .map(|(w, x)| w * x) // Given the weight and input value, multiply them.
                .sum::<f64>() //Sum all the products of the weighted inputs.
                + neuron.bias; // Add the bias to the neuron being calculated.
            neuron.z = z;
            neuron.output = activate(neuron.z, &neuron.activation); // optionally apply activation
            outputs.push(neuron.output);
        }
        //Return the vector corresponding to the neuron's outputs.
        outputs
    }
    //Support for downcasting Box<dyn LayerTrait> to InputLayer
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Structure to represent the composition of a neural network
struct NeuralNetwork {
    layers: Vec<Box<dyn LayerTrait>>,
    loss_fn: Loss,
}

impl NeuralNetwork {
    /// Create a new network
    /// `input_size` = number of input features
    /// `hidden_sizes` = number of neurons per hidden layer
    /// `output_size` = number of output neurons
    fn new(input_size: usize, hidden_layers: Vec<LayerConfig>, output_size: usize, output_activation: Activation, loss_fn: Loss) -> NeuralNetwork {
        // Create a vector of the layers that the Neural Network will contain
        let mut layers: Vec<Box<dyn LayerTrait>> = Vec::new();

        // Input layer
        // Initialize an InputLayer according the the indicated input_size.
        let input_layer:InputLayer = InputLayer { inputs: vec![0.0; input_size] };
        // Push (add) the input_layer to the Neural Network layers vector.
        layers.push(Box::new(input_layer));

        let mut prev_size = input_size;

        // Hidden layers
        //Cycle through the number of hidden_layers configurations in the passed vector
        for cfg in hidden_layers {
            // Initialize an empty vector to hold the neurons vector
            let mut neurons = Vec::new();
            // Retrieves a cryptographically secure, thread-local random number generator (RNG) that is lazily initialized and seeded by the system
            let mut rng = rand::thread_rng();

            // Cycle through the number of neurons indicated for the current hidden_layer configuration, cfg.
            for _ in 0..cfg.size {
                // Initialize a weights vector of the size of the previous layer, prev_size.
                // Initialize its values to some random value between -1 and 1.
                let weights: Vec<f64> = (0..prev_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
                // Initialize the bias values to some random value between -1 and 1.
                let bias = rng.gen_range(-1.0..1.0);
                neurons.push(Neuron {   //Push (add) a Neuron with the following initializations
                    weights,    // Randomly initialized weights vector of size prev_size
                    bias,       // Randomly initialized bias vector or prev_size
                    z: 0.0,     // Pre-activation output value set to 0
                    output: 0.0,// Post-activation output value set to 0
                    activation: cfg.activation.clone(), // Per-layer activation function
                });
            }

            // Push (add) the input_layer to the Neural Network layers vector.
            layers.push(Box::new(Layer { neurons }));
            // Set the new prev_size to the layer's size for the next layer's calculations.
            prev_size = cfg.size;
        }

        // Output layer
        // Initialize an empty vector to hold the neurons vector
        let mut neurons = Vec::new();
        // Retrieves a cryptographically secure, thread-local random number generator (RNG) that is lazily initialized and seeded by the system
        let mut rng = rand::thread_rng();

        // Cycle through the number of neurons indicated for the current hidden_layer configuration, cfg.
        for _ in 0..output_size {
            // Initialize a weights vector of the size of the previous layer, prev_size.
            // Initialize its values to some random value between -1 and 1.
            let weights: Vec<f64> = (0..prev_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
            // Initialize the bias values to some random value between -1 and 1.
            let bias = rng.gen_range(-1.0..1.0);
            neurons.push(Neuron {   //Push (add) a Neuron with the following initializations
                weights,    // Randomly initialized weights vector of size prev_size
                bias,       // Randomly initialized bias vector or prev_size
                z: 0.0,     // Pre-activation output value set to 0
                output: 0.0,// Post-activation output value set to 0
                activation: output_activation.clone(), // Per-layer activation function
            });
        }
        // Push (add) the input_layer to the Neural Network layers vector.
        layers.push(Box::new(Layer { neurons }));
        //Return the NeuralNetwork struct with the determined layers and provided loss_fn
        NeuralNetwork { layers, loss_fn }
    }

    // Forward pass through the entire network.
    // Determine the output of the Neural Network given some input.
    fn forward(&mut self, input_values: Vec<f64>) -> Vec<f64> {
        // Create an empty vector to store the activations (output) of each neuron
        let mut activations:Vec<f64>; //Could also use "Vec::new();"

        //Check if the input_values first layer is in fact of InputLayer type.
        if let Some(input_layer) = self.layers[0].as_any_mut().downcast_mut::<InputLayer>() {
            // Set input_values outputs as the input layer outputs to the Neural Network (self.layer[0])
            activations = input_layer.forward(input_values);
        }else {
            panic!("First layer must be of type InputLayer");  // Or return Err(...)
        };

        //From input layer to output layer, determine the output for each neuron.
        for layer in &mut self.layers[1..]{
            if let Some(hidden_layer) = layer.as_any_mut().downcast_mut::<Layer>() {
                // Set input_values outputs as the input layer outputs to the Neural Network (self.layer[0])
                activations = hidden_layer.forward(activations);
            }else {
                panic!("Non-input layer must be of type Layer");  // Or handle gracefully
            }
        };
        //Return the output of the output layer
        activations
    }

    fn forward_collect(&mut self, input_values: Vec<f64>) -> Vec<Vec<f64>> {
        let mut activations: Vec<Vec<f64>> = Vec::new();

        // Input layer
        let mut current = if let Some(input_layer) = self.layers[0].as_any_mut().downcast_mut::<InputLayer>() {
            input_layer.forward(input_values)
        } else {
            panic!("First layer must be of type InputLayer");
        };
        activations.push(current.clone());

        // Hidden + output layers (reuse forward logic)
        for layer in &mut self.layers[1..] {
            if let Some(l) = layer.as_any_mut().downcast_mut::<Layer>() {
                current = l.forward(current);
                activations.push(current.clone());
            }
        }

        activations
    }

    // Backpropagate through the network
    fn backward(&mut self, inputs: Vec<f64>, expected: Vec<f64>, lr: f64) {
        // ---- Forward pass (with activations) ----
        let activations = self.forward_collect(inputs);
        let outputs = activations.last().unwrap();
        let d_loss = loss_derivative(&expected, outputs, &self.loss_fn);

        let mut deltas: Vec<Vec<f64>> = vec![vec![]; self.layers.len()];

        // ---- Output layer deltas ----
        let output_index = self.layers.len() - 1;
        if let Some(output_layer) = self.layers[output_index].as_any_mut().downcast_mut::<Layer>() {
            deltas[output_index] = output_layer.neurons.iter().enumerate().map(|(j, neuron)| {
                d_loss[j] * activate_derivative(neuron.z, &neuron.activation)
            }).collect();
        }

        // ---- Hidden layer deltas ----
        for l in (1..output_index).rev() {
            // Split into two disjoint slices: [..=l] and [l+1..]
            let (left, right) = self.layers.split_at_mut(l + 1);

            let layer = left[l].as_any_mut().downcast_mut::<Layer>().unwrap();
            let next_layer = right[0].as_any_mut().downcast_mut::<Layer>().unwrap();

            deltas[l] = layer.neurons.iter().enumerate().map(|(j, neuron)| {
                let downstream: f64 = next_layer.neurons.iter().enumerate()
                    .map(|(k, next_neuron)| next_neuron.weights[j] * deltas[l + 1][k])
                    .sum();
                downstream * activate_derivative(neuron.z, &neuron.activation)
            }).collect();
        }

        // ---- Update weights and biases ----
        for l in 1..self.layers.len() {
            if let Some(layer) = self.layers[l].as_any_mut().downcast_mut::<Layer>() {
                for (j, neuron) in layer.neurons.iter_mut().enumerate() {
                    for (k, w) in neuron.weights.iter_mut().enumerate() {
                        *w -= lr * deltas[l][j] * activations[l - 1][k];
                    }
                    neuron.bias -= lr * deltas[l][j];
                }
            }
        }
    }

    fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)], epochs: usize, lr: f64, verbose: bool) {
        for epoch in 0..epochs {
            for (inputs, labels) in dataset {
                self.backward(inputs.clone(), labels.clone(), lr);
            }

            if verbose && epoch % 100 == 0 { // print every 100 epochs
                let mut total_loss = 0.0;
                for (x, y) in dataset {
                    let out = self.forward(x.clone());
                    total_loss += loss(y, &out, &self.loss_fn);
                }
                println!("Epoch {:4} | Loss = {:.6}", epoch, total_loss / dataset.len() as f64);
            }
        }

        if verbose {
            println!("Training complete.");
        }

        self.save("model.json");
    }

    fn test(&mut self, input: Vec<f64>, verbose: bool) -> Vec<f64> {
        let output = self.forward(input.clone());
        if verbose {
            println!("Test Input: {:?} -> Output: {:?}", input, output);
        }
        output
    }
    
    fn save(&self, path: &str) {
        // Convert layers to serializable form
        let mut serializable_layers = Vec::new();

        for layer in &self.layers[1..] { // immutable reference, okay
            // Use `as_any()` instead of `as_any_mut()`
            if let Some(l) = layer.as_any().downcast_ref::<Layer>() { 
                let neurons: Vec<SerializableNeuron> = l.neurons.iter().map(|n| SerializableNeuron {
                    weights: n.weights.clone(),
                    bias: n.bias,
                    activation: n.activation.clone(),
                }).collect();

                serializable_layers.push(SerializableLayer { neurons });
            }
        }

        let serializable_net = SerializableNetwork {
            layers: serializable_layers,
            loss_fn: self.loss_fn,
        };

        let file = File::create(path).expect("Failed to create file");
        serde_json::to_writer(file, &serializable_net).expect("Failed to write JSON");
    }

    #[allow(dead_code)]
    fn load(path: &str) -> Self {
        let file = File::open(path).expect("Failed to open file");
        let serializable_net: SerializableNetwork = serde_json::from_reader(file).expect("Failed to read JSON");

        let mut layers: Vec<Box<dyn LayerTrait>> = Vec::new();

        // Create input layer with size inferred from first serialized layer
        let input_size = serializable_net.layers.first().map(|l| l.neurons.first().unwrap().weights.len()).unwrap_or(0);
        layers.push(Box::new(InputLayer { inputs: vec![0.0; input_size] }));

        // Hidden and output layers
        for layer in serializable_net.layers {
            let neurons: Vec<Neuron> = layer.neurons.into_iter().map(|n| Neuron {
                weights: n.weights,
                bias: n.bias,
                z: 0.0,
                output: 0.0,
                activation: n.activation,
            }).collect();

            layers.push(Box::new(Layer { neurons }));
        }

        NeuralNetwork {
            layers,
            loss_fn: serializable_net.loss_fn,
        }
    }
}

fn main() {
    let start = Instant::now();

    //Select which test to run
    // test_xor();
    // test_xor_noise();
    // test_3bit_parity();
    // test_multi_xor();
    test_sin_approximation();
    let elapsed = start.elapsed();
    eprintln!("\nProgram execution time: {:.2}s", elapsed.as_secs_f64());
}

fn test_xor(){
    // 1. Define XOR dataset
    let dataset = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // 2. Build network: 2 inputs -> [3 hidden neurons] -> 1 output
    let mut net = NeuralNetwork::new(
        2,
        vec![LayerConfig { size: 3, activation: Activation::Sigmoid }],
        1,
        Activation::Sigmoid,
        Loss::MeanSquaredError,
    );

    // 3. Train network with verbose logging
    println!("Training XOR network...");
    net.train(&dataset, 5000, 0.5, true);

    // 4. Test network with verbose logging
    println!("\nTesting XOR network:");
    for (x, y_true) in &dataset {
        let y_pred = net.test(x.clone(), true);
        println!(
            "Expected: {:?}, Predicted (rounded): {}",
            y_true,
            if y_pred[0] > 0.5 { 1 } else { 0 }
        );
    }
}

fn test_xor_noise() {
    // XOR with small noise
    let dataset = vec![
        (vec![0.0 + 0.1, 0.0 - 0.1], vec![0.0]),
        (vec![0.0 - 0.1, 1.0 + 0.05], vec![1.0]),
        (vec![1.0 + 0.05, 0.0 - 0.05], vec![1.0]),
        (vec![1.0 - 0.05, 1.0 + 0.1], vec![0.0]),
    ];

    let mut net = NeuralNetwork::new(
        2,
        vec![
            LayerConfig { size: 5, activation: Activation::Sigmoid },
            LayerConfig { size: 3, activation: Activation::Sigmoid },
        ],
        1,
        Activation::Sigmoid,
        Loss::MeanSquaredError,
    );

    println!("Training XOR with noise network...");
    net.train(&dataset, 5000, 0.5, true);

    println!("\nTesting XOR with noise network:");
    for (x, y_true) in &dataset {
        let y_pred = net.test(x.clone(), true);
        println!(
            "Expected: {:?}, Predicted (rounded): {}",
            y_true,
            if y_pred[0] > 0.5 { 1 } else { 0 }
        );
    }
}

fn test_3bit_parity() {
    // 3-bit parity
    let dataset = vec![
        (vec![0.0,0.0,0.0], vec![0.0]),
        (vec![0.0,0.0,1.0], vec![1.0]),
        (vec![0.0,1.0,0.0], vec![1.0]),
        (vec![0.0,1.0,1.0], vec![0.0]),
        (vec![1.0,0.0,0.0], vec![1.0]),
        (vec![1.0,0.0,1.0], vec![0.0]),
        (vec![1.0,1.0,0.0], vec![0.0]),
        (vec![1.0,1.0,1.0], vec![1.0]),
    ];

    let mut net = NeuralNetwork::new(
        3,
        vec![
            LayerConfig { size: 6, activation: Activation::Sigmoid },
            LayerConfig { size: 4, activation: Activation::Sigmoid },
        ],
        1,
        Activation::Sigmoid,
        Loss::MeanSquaredError,
    );

    println!("Training 3-bit parity network...");
    net.train(&dataset, 10000, 0.5, true);

    println!("\nTesting 3-bit parity network:");
    for (x, y_true) in &dataset {
        let y_pred = net.test(x.clone(), true);
        println!(
            "Expected: {:?}, Predicted (rounded): {}",
            y_true,
            if y_pred[0] > 0.5 { 1 } else { 0 }
        );
    }
}

fn test_sin_approximation() {
    // Function approximation: y = sin(x)
    let dataset: Vec<(Vec<f64>, Vec<f64>)> = (0..100).map(|i| {
        let x = i as f64 * 2.0 * std::f64::consts::PI / 100.0;
        (vec![x], vec![x.sin()])
    }).collect();

    let mut net = NeuralNetwork::new(
        1,
        vec![
            LayerConfig { size: 20, activation: Activation::Tanh },
            LayerConfig { size: 20, activation: Activation::Tanh },
        ],
        1,
        Activation::Identity,
        Loss::MeanSquaredError,
    );

    println!("Training sin(x) approximation network...");
    net.train(&dataset, 10000, 0.01, true);

    println!("\nTesting sin(x) approximation network:");
    for (x, y_true) in &dataset {
        let y_pred = net.test(x.clone(), true);
        println!(
            "x: {:.2}, Expected: {:.4}, Predicted: {:.4}",
            x[0], y_true[0], y_pred[0]
        );
    }
}

fn test_multi_xor() {
    // Two XORs combined
    let dataset = vec![
        (vec![0.0,0.0,0.0,0.0], vec![0.0,0.0]),
        (vec![0.0,1.0,1.0,0.0], vec![1.0,1.0]),
        (vec![1.0,0.0,0.0,1.0], vec![1.0,1.0]),
        (vec![1.0,1.0,1.0,1.0], vec![0.0,0.0]),
    ];

    let mut net = NeuralNetwork::new(
        4,
        vec![
            LayerConfig { size: 6, activation: Activation::Sigmoid },
            LayerConfig { size: 4, activation: Activation::Sigmoid },
        ],
        2,
        Activation::Sigmoid,
        Loss::MeanSquaredError,
    );

    println!("Training multi-XOR network...");
    net.train(&dataset, 10000, 0.5, true);

    println!("\nTesting multi-XOR network:");
    for (x, y_true) in &dataset {
        let y_pred = net.test(x.clone(), true);
        let y_rounded: Vec<u8> = y_pred.iter().map(|v| if *v > 0.5 { 1 } else { 0 }).collect();
        println!(
            "Expected: {:?}, Predicted (rounded): {:?}",
            y_true, y_rounded
        );
    }
}

#[allow(dead_code)]
fn plot_comparison (){
    // Example: generate fake dataset (y = 2x + noise)
    let n = 100;
    let mut rng = rand::thread_rng();
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    //Linear Plot
    // let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + rng.gen_range(-1.0..1.0)).collect();
    //Hyperbolic plot
    let y: Vec<f64> = x.iter().map(|&xi| xi.powi(2) + rng.gen_range(-1.0..1.0)).collect();

    // Scatter plot of data points
    let trace_data = Scatter::new(x.clone(), y.clone())
        .mode(Mode::Markers)
        .name("Training Data");

    // Example "prediction line" (pretend a model output)
    //Linear Plot
    // let predicted: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
    //Hyperbolic Plot
    let predicted: Vec<f64> = x.iter().map(|&xi| xi.powi(2)).collect();
    let trace_model = Scatter::new(x, predicted)
        .mode(Mode::Lines)
        .name("Model Prediction");

    // Build plot
    let mut plot = Plot::new();
    plot.add_trace(trace_data);
    plot.add_trace(trace_model);

    plot.set_layout(
        plotly::Layout::new()
            .title(Title::new("Neural Network Sandbox: Data vs Model")),
    );

    // Save to HTML file
    plot.write_html("output/plot.html");
    println!("Plot written to output/plot.html");
}

#[allow(dead_code)]
fn load_training_data(path: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut dataset = Vec::new();
    for entry in glob(&format!("{}/*.csv", path)).expect("Failed to read glob pattern") {
        if let Ok(file_path) = entry {
            let file = File::open(&file_path).unwrap();
            let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

            for result in rdr.records() {
                let record = result.unwrap();
                let values: Vec<f64> = record.iter().map(|x| x.parse::<f64>().unwrap()).collect();

                // Example: last value is label, rest are inputs
                let (input, label) = values.split_at(values.len() - 1);
                dataset.push((input.to_vec(), label.to_vec()));
            }
        }
    }
    dataset
}

#[derive(Serialize, Deserialize)]
struct SerializableNeuron {
    weights: Vec<f64>,
    bias: f64,
    activation: Activation,
}

#[derive(Serialize, Deserialize)]
struct SerializableLayer {
    neurons: Vec<SerializableNeuron>,
}

#[derive(Serialize, Deserialize)]
struct SerializableNetwork {
    layers: Vec<SerializableLayer>,
    loss_fn: Loss,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Activation functions ---
    #[test]
    fn test_activation_sigmoid() {
        let x = 0.0;
        let y = activate(x, &Activation::Sigmoid);
        assert!((y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_activation_relu() {
        assert_eq!(activate(-1.0, &Activation::Relu), 0.0);
        assert_eq!(activate(2.0, &Activation::Relu), 2.0);
    }

    #[test]
    fn test_activation_derivative_tanh() {
        let x = 0.5;
        let d = activate_derivative(x, &Activation::Tanh);
        let manual = 1.0 - x.tanh().powi(2);
        assert!((d - manual).abs() < 1e-6);
    }

    // --- Loss functions ---
    #[test]
    fn test_loss_mse() {
        let y_true = vec![1.0, 0.0];
        let y_pred = vec![0.5, 0.5];
        let l = loss(&y_true, &y_pred, &Loss::MeanSquaredError);
        assert!((l - 0.25).abs() < 1e-6); // ((1-0.5)^2 + (0-0.5)^2)/2
    }

   #[test]
    fn test_loss_derivative_mse() {
        let y_true = vec![1.0];
        let y_pred = vec![0.5];
        let d = loss_derivative(&y_true, &y_pred, &Loss::MeanSquaredError);
        assert!((d[0] - (-1.0)).abs() < 1e-6); // matches your implementation
    }

    // --- Layer forward pass ---
    #[test]
    fn test_layer_forward_identity() {
        let mut layer = Layer {
            neurons: vec![
                Neuron {
                    weights: vec![1.0, 1.0],
                    bias: 0.0,
                    z: 0.0,
                    output: 0.0,
                    activation: Activation::Identity,
                }
            ]
        };
        let input = vec![2.0, 3.0];
        let output = layer.forward(input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 5.0); // 2*1 + 3*1 + 0 bias
    }

    // --- Network forward pass ---
    #[test]
    fn test_network_forward() {
        let mut net = NeuralNetwork::new(
            2,
            vec![LayerConfig { size: 2, activation: Activation::Sigmoid }],
            1,
            Activation::Sigmoid,
            Loss::MeanSquaredError,
        );
        let output = net.forward(vec![0.0, 0.0]);
        assert_eq!(output.len(), 1);
    }

    // --- Backpropagation weight update ---
   #[test]
    fn test_backprop_updates_weights() {
        // Build a network with deterministic weights
        let mut net = NeuralNetwork {
            layers: vec![
                Box::new(InputLayer { inputs: vec![0.0, 0.0] }),
                Box::new(Layer {
                    neurons: vec![
                        Neuron {
                            weights: vec![0.5, -0.5],
                            bias: 0.0,
                            z: 0.0,
                            output: 0.0,
                            activation: Activation::Sigmoid,
                        }
                    ],
                }),
                Box::new(Layer {
                    neurons: vec![
                        Neuron {
                            weights: vec![0.5],
                            bias: 0.0,
                            z: 0.0,
                            output: 0.0,
                            activation: Activation::Sigmoid,
                        }
                    ],
                }),
            ],
            loss_fn: Loss::MeanSquaredError,
        };

        let inputs = vec![1.0, 0.0];   // ensures non-zero activation
        let target = vec![0.0];        // ensures non-zero error

        // Capture weight before
        let old_weight = if let Some(layer) = net.layers[2].as_any().downcast_ref::<Layer>() {
            layer.neurons[0].weights[0]
        } else {
            panic!("Layer cast failed");
        };

        net.backward(inputs.clone(), target.clone(), 0.1);

        // Capture weight after
        let new_weight = if let Some(layer) = net.layers[2].as_any().downcast_ref::<Layer>() {
            layer.neurons[0].weights[0]
        } else {
            panic!("Layer cast failed");
        };

        assert!(
            (old_weight - new_weight).abs() > 1e-8,
            "Weight should be updated"
        );
    }

    // --- Training loop ---
    #[test]
    fn test_training_reduces_loss() {
        let mut net = NeuralNetwork::new(
            2,
            vec![LayerConfig { size: 3, activation: Activation::Sigmoid }],
            1,
            Activation::Sigmoid,
            Loss::MeanSquaredError,
        );

        // Simple dataset: XOR
        let dataset = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        // Compute initial loss
        let mut total_loss = 0.0;
        for (x, y) in &dataset {
            let out = net.forward(x.clone());
            total_loss += loss(y, &out, &Loss::MeanSquaredError);
        }

        // Train
        net.train(&dataset, 50, 0.5, true);

        // Compute loss after training
        let mut total_loss_after = 0.0;
        for (x, y) in &dataset {
            let out = net.forward(x.clone());
            total_loss_after += loss(y, &out, &Loss::MeanSquaredError);
        }

        assert!(total_loss_after < total_loss, "Training should reduce loss");
    }
}
