import * as tfvis from "@tensorflow/tfjs-vis"
import * as tf from "@tensorflow/tfjs"

window.onload =async function(){
    const xs = [1,2,3,4]
    const ys = [1,3,5,7]
    const predictInput = document.getElementById("predictInput")
    const predictBtn = document.getElementById("predictBtn")

    const model = tf.sequential();
    model.add(tf.layers.dense({units:1, inputShape:[1]}))
    model.compile({ loss:tf.losses.meanSquaredError, optimizer:tf.train.sgd(0.1) })
    const inputs = tf.tensor(xs)
    const labels = tf.tensor(ys)
    await model.fit(inputs, labels,
        {
            batchSize:1,
            epochs:100,
            callbacks:tfvis.show.fitCallbacks(
                {name : "训练过程"},
                ["loss"]
            )
        }
    )
    predictBtn.classList.remove("hide")
    predictBtn.onclick = function(){
        const output = model.predict(tf.tensor([Number(predictInput.value)]))
        console.log(output.dataSync()[0])
    }
}