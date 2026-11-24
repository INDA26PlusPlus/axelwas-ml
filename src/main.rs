use mnist::*;
use ndarray::prelude::*;

type Matrix<'a> = ArrayBase<ndarray::ViewRepr<&'a f32>, Dim<[usize; 2]>, f32>;

fn relu(val: f32) -> f32 {
    if val > 0.0 {
        return val;
    }

    val * -0.125
}

fn predict(shape: Matrix, parameters: (Matrix, Matrix)) -> [f32; 10] {
    let product = shape.dot(&parameters.0);
    let relued = product.map(|v| relu(*v));
    let result = relued.dot(&parameters.1);
    
    todo!();
}

fn randomParameters() -> (Matrix, Matrix) {
    const SIZE = 28 * 28;

    
    
}


fn main() {
    let l: ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 2]>, f32> = ArrayView::from_shape((3, 1), &[1.0_f32, 2.0, 3.0]).unwrap();

    let m: ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 2]>, f32> = ArrayView::from_shape((3, 3), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let res = m.dot(&l);

    println!("{:?}", m.dot(&l));
    println!("{:?}", res.get((0, 0)));

    // println!("{:?}", s.dot(&l));

    // println!("{:?}", s);

    return;
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

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));



    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);
    
}