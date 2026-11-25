use core::f32;
use rayon::prelude::*;

use mnist::*;
use ndarray::prelude::*;

type Matrix = ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>, f32>;

type MatrixBorrow<'a> = ArrayBase<ndarray::ViewRepr<&'a f32>, Dim<[usize; 2]>, f32>;
type Model = (Matrix, Matrix);
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
    let product = parameters.0.dot(&shape);
    let relued = product.map(|v| relu(*v));
    let result = parameters.1.dot(&relued);
    
    let mut iter = result.into_iter();
    let raw = [iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()];
    softmax(raw)
}

fn mutate(parent: &Model, delta: f32) -> Model {
    (parent.0.map(|v| *v - rand::random_range(-delta..delta)), parent.1.map(|v| *v - rand::random_range(-delta..delta)))
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
    
    (big, small)
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

fn best_model(models: &[Model], data: Data) -> usize {
    let mut index = 0;
    let mut loss = 1.0;

    let after: Vec<(usize, f32)> = models.par_iter().enumerate().map(|(i, m)| (i, calculate_loss(m, data, 25000))).collect();

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
    for _ in 0..1000 {
        let index_best = best_model(&models, &combined);
        let best = &models[index_best];
        best_clone = models[index_best].clone();
        let now_loss = calculate_loss(&best, &combined, 25000);
        println!("{}", now_loss);

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

    println!("loss on test images: {}", calculate_loss(&best_clone, &combined, 10000));
    
}