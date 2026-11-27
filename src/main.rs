use core::f32;
use std::{ops::Add, os::unix::raw::nlink_t};
use rayon::prelude::*;

use mnist::*;
use ndarray::prelude::*;

type Matrix = ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>, f32>;

type MatrixBorrow<'a> = ArrayBase<ndarray::ViewRepr<&'a f32>, Dim<[usize; 2]>, f32>;

#[derive(Debug, Clone)]
struct Model {
    first_layer: Matrix,
    bias: Matrix,
    final_layer: Matrix
}
type Data<'a> = &'a [(&'a [f32], usize)];

fn relu(val: f32) -> f32 {
    if val > 0.0 {
        return val;
    }

    val * -0.125
}

fn softmax(nums: [f32; 10]) -> [f32; 10] {
    let mut max = -100000000.0;
    for n in &nums {
        if max < *n {
            max = *n;
        }
    }

    let exps = nums.into_iter().map(|v| (v - max).exp());
    let total: f32 = exps.clone().sum();
    let mut ret = exps.map(|v| v / total);

    [ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap(), ret.next().unwrap()]
}

fn predict(shape: MatrixBorrow, parameters: &Model) -> [f32; 10] {
    let product = parameters.first_layer.dot(&shape);
    let product = product.add(&parameters.bias);
    let relued = product.map(|v| relu(*v));
    let result = parameters.final_layer.dot(&relued);
    
    let mut iter = result.into_iter();
    let raw = [iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()];
    softmax(raw)
}

fn mutate(parent: &Model, delta: f32) -> Model {
    Model { 
        first_layer: parent.first_layer.map(|v| *v - rand::random_range(-delta..delta)), 
        bias: parent.bias.map(|v| *v - rand::random_range(-delta..delta)), 
        final_layer: parent.final_layer.map(|v| *v - rand::random_range(-delta..delta)) 
    }
}

fn random_parameters() -> Model {
    const SIZE: usize = 28 * 28;

    let generator = || rand::random_range(-1.1..1.1);
    let mut first_matrix = vec![];
    for _ in 0..(SIZE * 28) {
        first_matrix.push(generator());
    }

    let big = Array2::from_shape_vec((28, SIZE), first_matrix).unwrap();

    let mut second_matrix = vec![];
    for _ in 0..(28 * 10) {
        second_matrix.push(generator());
    }  
    let small = Array2::from_shape_vec((10, 28), second_matrix).unwrap();
    
    let mut bias = vec![];

    for _ in 0..(28) {
        bias.push(generator());
    }  

    let bias = Array2::from_shape_vec((28, 1), bias).unwrap();

    Model {
        first_layer: big,
        final_layer: small,
        bias: bias
    }
}


fn print_image(img: Vec<f32>) {
    for (i, item) in img.iter().enumerate() {
        if i % 28 == 0 {
            println!();
        }
        print!("{}", match item {
            0.0..0.1 => ' ',
            0.1..0.7 => 'o',
            0.7..=1.0 => '#',
            _ => panic!()
        });
    }
}

fn calculate_loss(model: &Model, examples: Data, until: usize) -> f32 {

    let mut sum: f64 = 0.0;
    for n in 0..until {
        let example_n = n;//rand::random_range(0..examples.len());
        let pic = ArrayView::from_shape((28 * 28, 1), examples[example_n].0).unwrap();
        
        let res = predict(pic, &model);
        let val = 1.0 - res[examples[example_n].1];
        sum += val as f64;
    }

    (sum / until as f64) as f32


}

fn calculate_right(model: &Model, examples: Data, until: usize) -> f32 {

    let mut sum: usize = 0;
    for n in 0..until {
        let example_n = n;//rand::random_range(0..examples.len());
        let pic = ArrayView::from_shape((28 * 28, 1), examples[example_n].0).unwrap();
        
        let res = predict(pic, &model);
        let mut val_max = -1.0;
        let mut index_max = 0;
        for (i, n) in res.iter().enumerate() {
            if *n > val_max {
                val_max = *n;
                index_max = i;
            }
        }

        if index_max == examples[example_n].1 {
            sum = sum + 1;
        }
    }

    (sum as f64 / until as f64) as f32


}

fn train_data_into_vec(trn_img: Vec<u8>, length: usize) -> Vec<Vec<f32>> {
    let mut ret = vec![];
    for n in 0..length {
        let mut img = vec![];
        for m in 0..(28 * 28) {
            img.push(trn_img[n * 28 * 28 + m] as f32 / 256.0);
        }
        ret.push(img);
    }
    ret
}

fn best_model(models: &[Model], data: Data, until: usize) -> usize {
    let mut index = 0;
    let mut loss = 1.0;

    let after: Vec<(usize, f32)> = models.par_iter().enumerate().map(|(i, m)| (i, calculate_loss(m, data, until))).collect();

    for (i, r) in after {
        if r < loss {
            loss = r;
            index = i;
        }
    }

    return index;
}

fn main() {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = train_data_into_vec(trn_img, 50_000);



    // Convert the returned Mnist struct to Array2 format
    let train_labels: Vec<usize> = trn_lbl
        .iter()
        .map(|x| *x as usize)
        .collect();
    
    let combined: Vec<_> = train_data.iter().zip(train_labels).map(|s| (s.0.as_slice(), s.1)).collect();

    let mut models: [Model; 200] = std::array::from_fn(|_| random_parameters());
    let mut best_clone = models[0].clone();

    let mut last_loss = 1.0;
    let mut diff = 0.2;
    let mut until = 100;
    for i in 0..1100 {
        if i % 10 == 0 {
            until = (until as f64 * (25000.0 - 100.0 as f64).powf(1.0/110.0 as f64) as f64) as usize;
            if until >= 50000 {
                until = 49999;
            }
        }
        let index_best = best_model(&models, &combined, until);
        let best = &models[index_best];
        best_clone = models[index_best].clone();
        let now_loss = calculate_loss(&best, &combined, until);
        println!("{} nr {}", now_loss, i);

        if now_loss == last_loss {
            println!("evolution is now slower");
            diff = diff * 0.98;
        }
        last_loss = now_loss;

        models = std::array::from_fn(|i| if i == 0 { best.clone() } else { mutate(&best, 0.1)});

    }
    
    let test_data = train_data_into_vec(tst_img, 10_000);
    let test_labels: Vec<usize> = tst_lbl
        .iter()
        .map(|x| *x as usize)
        .collect();

    let combined: Vec<_> = test_data.iter().zip(test_labels).map(|s| (s.0.as_slice(), s.1)).collect();

    println!("loss on test images: {}. Correctness: {}", calculate_loss(&best_clone, &combined, 10000), calculate_right(&best_clone, &combined, 10000));
    
}