import tf from '@tensorflow/tfjs';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Camada 1: 7 entrada → 80 neurônios (ReLU)
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Camada 2: 80 neurônios → 3 saídas (Softmax para probabilidades)
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilação: Adam optimizer + Categorical Crossentropy loss
    model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Treino: 100 épocas, shuffle para evitar overfitting
    await model.fit(inputXs, outputYs, { 
        verbose: 0,
        epochs: 100, 
        shuffle: true, 
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    return model;
}

async function predict(model, pessoa) {
    const tfInput = tf.tensor2d(pessoa);
    const prediction = model.predict(tfInput);
    const predArray = await prediction.array();
    return predArray[0].map((prob, index) => ({prob, index}))
}

// Dados normalizados e one-hot encoded: [idade, azul, vermelho, verde, SP, Rio, Curitiba]
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels: [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"];
const tensorLabels = [
    [1, 0, 0], // Erick → premium
    [0, 1, 0], // Ana → medium
    [0, 0, 1]  // Carlos → basic
];

const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

const pessoa = {nome: 'zé', idade: 28, cor: 'verde', localização: "Curitiba"}

// Normalização da idade 
// idade_min - 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2

const pessoaTensorNormalizado = [
    [
      0.2, // idade normalizada
      1,   // azul
      0,   // vermelho  
      0,   // verde
      0,   // SP
      0,   // Rio
      0    // Curitiba  
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')

console.log(results)