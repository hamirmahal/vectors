#[derive(Debug)]
struct Vector<const N: usize>([f64; N]);
impl Vector<3> {
    fn cross(&self, b: &Vector<3>) -> Vector<3> {
        Vector([
            self.0[1] * b.0[2] - self.0[2] * b.0[1],
            self.0[2] * b.0[0] - self.0[0] * b.0[2],
            self.0[0] * b.0[1] - self.0[1] * b.0[0],
        ])
    }
}
impl<const N: usize> Vector<N> {
    fn dot(&self, v: &Vector<N>) -> f64 {
        self.0.iter().zip(v.0.iter()).map(|(a, b)| a * b).sum()
    }
}

fn main() {
    let f1 = Vector([10.0, -20.4, 2.0]);
    let f2 = Vector([-15.0, 0.0, -6.2]);
    let result = quadratic_equation(9.0, -126.0, 441.0);
    let the_angle_between_f1_and_f2 = the_angle_between(&f1, &f2);
    let centripetal_acceleration = get_centripetal_acceleration(20.0, 14.9);
    let v_x = cos(29.0) * 17000.0;
    let v_y = sin(29.0) * 17000.0;
    dbg!(v_x);
    dbg!(v_y);
    println!("The magnitude of f1 is {}", the_magnitude_of(&f1));
    println!("The magnitude of f2 is {}", the_magnitude_of(&f2));
    println!("The roots of the equation are {:?}", result);
    println!("f1 cross f2 is {:?}", f1.cross(&f2));
    println!(
        "The angle between f1 and f2 is {} degrees",
        the_angle_between_f1_and_f2
    );
    println!(
        "The centripetal acceleration is {}",
        centripetal_acceleration
    );
}

/// This function returns the cosine of an angle in degrees.
fn cos(x: f64) -> f64 {
    x.to_radians().cos()
}
fn get_centripetal_acceleration(v: f64, r: f64) -> f64 {
    v.powi(2) / r
}
fn quadratic_equation(a: f64, b: f64, c: f64) -> (f64, f64) {
    (
        (-b + f64::sqrt(b.powi(2) - 4.0 * a * c)) / (2.0 * a),
        (-b - f64::sqrt(b.powi(2) - 4.0 * a * c)) / (2.0 * a),
    )
}
/// This function returns the inverse cosine of a number in degrees.
fn inverse_cosine(x: f64) -> f64 {
    x.acos().to_degrees()
}
/// This function returns the sine of an angle in degrees.
fn sin(x: f64) -> f64 {
    x.to_radians().sin()
}
fn the_magnitude_of<const N: usize>(v: &Vector<N>) -> f64 {
    v.0.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}
/// This function returns the angle between two vectors in degrees.
fn the_angle_between<const N: usize>(a: &Vector<N>, b: &Vector<N>) -> f64 {
    let cos_of_angle = a.dot(b) / (the_magnitude_of(a) * the_magnitude_of(b));
    inverse_cosine(cos_of_angle)
}

#[cfg(test)]
mod tests {
    use super::*;
    const V1: Vector<3> = Vector([10.0, -20.4, 2.0]);
    const V2: Vector<3> = Vector([-15.0, 0.0, -6.2]);

    #[test]
    fn test_cross() {
        let perpendicular_vectors = [Vector([3.0, -1.0, 4.0]), Vector([7.0, 1.0, -5.0])];
        let there_is_a_floating_point_discrepancy = the_magnitude_of(&perpendicular_vectors[0])
            * the_magnitude_of(&perpendicular_vectors[1])
            > the_magnitude_of(&perpendicular_vectors[0].cross(&perpendicular_vectors[1]));
        let the_magnitude_of_the_cross_product_of_perpendicular_vectors_is_the_product_of_their_magnitudes =
            (the_magnitude_of(&perpendicular_vectors[0])
                * the_magnitude_of(&perpendicular_vectors[1])
                - the_magnitude_of(&perpendicular_vectors[0].cross(&perpendicular_vectors[1])))
                < 0.00000000000001;
        assert!(the_magnitude_of_the_cross_product_of_perpendicular_vectors_is_the_product_of_their_magnitudes);
        assert!(there_is_a_floating_point_discrepancy);
    }
    #[test]
    fn test_centripetal_acceleration() {
        assert_eq!(get_centripetal_acceleration(20.0, 14.9), 26.845637583892618);
    }
    #[test]
    fn test_cos() {
        assert_eq!(cos(29.0) * 17000.0, 14868.535021369727);
        assert_eq!(cos(30.0), 0.8660254037844387);
        assert_eq!(cos(45.0), std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(cos(60.0), 0.5000000000000001);
    }
    #[test]
    fn test_dot() {
        assert_eq!(V1.dot(&V2), -162.4);
    }
    #[test]
    fn test_quadratic_equation() {
        assert_eq!(quadratic_equation(9.0, -126.0, 441.0), (7.0, 7.0));
    }
    #[test]
    fn test_sin() {
        assert_eq!(sin(29.0) * 17000.0, 8241.76354418773);
        assert_eq!(sin(30.0), 0.49999999999999994);
        assert_eq!(sin(45.0), 0.7071067811865475);
        assert_eq!(sin(60.0), f64::sqrt(3.0) / 2.0);
    }
    #[test]
    fn test_the_magnitude_of() {
        assert_eq!(the_magnitude_of(&V1), 22.807016464237492);
        assert_eq!(the_magnitude_of(&V2), 16.230834852218784);
        assert_eq!(the_magnitude_of(&Vector([-7.0, 9.0])), 11.40175425099138);
        assert_eq!(the_magnitude_of(&Vector([-3.0, 5.0])), 5.830951894845301);
        assert_eq!(the_magnitude_of(&Vector([10.0, 23.0])), 25.079872407968907);
    }
    #[test]
    fn test_the_angle_between() {
        assert_eq!(the_angle_between(&V1, &V2), 116.02154864365895);
    }
}
