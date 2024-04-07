#[derive(Debug, PartialEq)]
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
impl<const N: usize> std::convert::AsRef<Vector<N>> for Vector<N> {
    fn as_ref(&self) -> &Vector<N> {
        self
    }
}
impl<const N: usize> std::fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();
        self.0.iter().enumerate().for_each(|(i, &x)| {
            if x != 0.0 {
                result.push_str(if result.is_empty() { "" } else { " " });
                if x < 0.0 {
                    result.push_str("- ");
                } else if !result.is_empty() {
                    result.push_str("+ ");
                }
                if x.abs() != 1.0 {
                    result.push_str(&x.abs().to_string());
                }
                result.push((b'i' + i as u8) as char);
            }
        });
        write!(f, "{}", if result.is_empty() { "0i" } else { &result })
    }
}
impl<'a, const N: usize> std::ops::Add<&'a Vector<N>> for &'a Vector<N> {
    type Output = Vector<N>;
    fn add(self, rhs: &'a Vector<N>) -> Vector<N> {
        Vector(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}
impl<'a, const N: usize> std::ops::Add<Vector<N>> for &'a Vector<N> {
    type Output = Vector<N>;
    fn add(self, rhs: Vector<N>) -> Vector<N> {
        self + &rhs
    }
}
impl<'a, const N: usize> std::ops::Add<&'a Vector<N>> for Vector<N> {
    type Output = Self;
    fn add(self, rhs: &'a Vector<N>) -> Self {
        &self + rhs
    }
}
impl<const N: usize> std::ops::Add for Vector<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        &self + &rhs
    }
}

impl<const N: usize> std::ops::Div<f64> for Vector<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Vector(
            self.0
                .iter()
                .map(|&x| x / rhs)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}
impl<const N: usize> std::ops::Mul<f64> for Vector<N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Vector(
            self.0
                .iter()
                .map(|&x| x * rhs)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}
impl<const N: usize> std::ops::Mul<Vector<N>> for f64 {
    type Output = Vector<N>;
    fn mul(self, rhs: Vector<N>) -> Vector<N> {
        rhs * self
    }
}

fn main() {
    let m = [0.02; 6];
    let r = [25, 15, 5, 5, 15, 25].map(|x| x as f64 / 100.0);
    let f1 = Vector([10.0, -20.4, 2.0]);
    let f2 = Vector([-15.0, 0.0, -6.2]);
    let result = quadratic_equation(9.0, -126.0, 441.0);
    let inertia = the_moment_of_inertia_of(&m, &r);
    let the_angle_between_f1_and_f2 = the_angle_between(&f1, &f2);
    let centripetal_acceleration = get_centripetal_acceleration(20.0, 14.9);
    let v_x = cos(29.0) * 17000.0;
    let v_y = sin(29.0) * 17000.0;
    let i = inverse_tan(2.0);
    let t = tan(i);
    dbg!(t);
    dbg!(v_x);
    dbg!(v_y);
    dbg!(sqrt(3.0));
    println!("The magnitude of f1 is {}", the_magnitude_of(&f1));
    println!("The magnitude of f2 is {}", the_magnitude_of(&f2));
    println!("The roots of the equation are {:?}", result);
    println!("The moment of inertia is {}", inertia);
    println!("f1 cross f2 is {}", f1.cross(&f2));
    println!(
        "The angle between f1 and f2 is {} degrees",
        the_angle_between_f1_and_f2
    );
    println!(
        "The centripetal acceleration is {}",
        centripetal_acceleration
    );
}

fn cos(degrees: impl Into<f64>) -> f64 {
    degrees.into().to_radians().cos()
}
fn get_centripetal_acceleration(v: impl Into<f64>, r: impl Into<f64>) -> f64 {
    v.into().powi(2) / r.into()
}
fn quadratic_equation(a: impl Into<f64>, b: impl Into<f64>, c: impl Into<f64>) -> (f64, f64) {
    let a = a.into();
    let b = b.into();
    let c = c.into();
    (
        (-b + f64::sqrt(b.powi(2) - 4.0 * a * c)) / (2.0 * a),
        (-b - f64::sqrt(b.powi(2) - 4.0 * a * c)) / (2.0 * a),
    )
}
/// This function returns the inverse cosine of a number in degrees.
fn inverse_cosine(x: f64) -> f64 {
    x.acos().to_degrees()
}
/// This function returns the inverse tangent of a number in degrees.
fn inverse_tan(x: f64) -> f64 {
    x.atan().to_degrees()
}
fn sin(degrees: impl Into<f64>) -> f64 {
    degrees.into().to_radians().sin()
}
fn sqrt(x: impl Into<f64>) -> f64 {
    x.into().sqrt()
}
fn tan(degrees: impl Into<f64>) -> f64 {
    degrees.into().to_radians().tan()
}
fn the_magnitude_of<const N: usize>(v: impl AsRef<Vector<N>>) -> f64 {
    f64::sqrt(v.as_ref().0.iter().map(|x| x.powi(2)).sum())
}
fn the_moment_of_inertia_of<const N: usize>(masses_in_kg: &[f64; N], radii_in_m: &[f64; N]) -> f64 {
    masses_in_kg
        .iter()
        .zip(radii_in_m.iter())
        .map(|(&m, &r)| m * r.powi(2))
        .sum()
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
    fn verify<const N: usize>(expected_output: &[(Vector<N>, &str)]) {
        expected_output
            .iter()
            .for_each(|(v, expected)| assert_eq!(format!("{}", v), *expected));
    }

    #[test]
    fn test_cross() {
        let perpendicular_vectors = [Vector([3.0, -1.0, 4.0]), Vector([7.0, 1.0, -5.0])];
        let there_is_a_floating_point_discrepancy = the_magnitude_of(&perpendicular_vectors[0])
            * the_magnitude_of(&perpendicular_vectors[1])
            > the_magnitude_of(perpendicular_vectors[0].cross(&perpendicular_vectors[1]));
        let the_magnitude_of_the_cross_product_of_perpendicular_vectors_is_the_product_of_their_magnitudes =
            (the_magnitude_of(&perpendicular_vectors[0])
                * the_magnitude_of(&perpendicular_vectors[1])
                - the_magnitude_of(perpendicular_vectors[0].cross(&perpendicular_vectors[1])))
                < 0.00000000000001;
        assert!(the_magnitude_of_the_cross_product_of_perpendicular_vectors_is_the_product_of_their_magnitudes);
        assert!(there_is_a_floating_point_discrepancy);
    }
    #[test]
    fn test_centripetal_acceleration() {
        let r = 73.0;
        let rev_per_s = 0.5;
        let v = rev_per_s * 2.0 * std::f64::consts::PI * r;
        assert_eq!(get_centripetal_acceleration(25, 500), 1.25);
        assert_eq!(get_centripetal_acceleration(25.0, 500.0), 1.25);
        assert_eq!(get_centripetal_acceleration(v, r), 720.481121279523);
        assert_eq!(get_centripetal_acceleration(20, 14.9), 26.845637583892618);
        assert_eq!(get_centripetal_acceleration(20.0, 14.9), 26.845637583892618);
    }
    #[test]
    fn test_cos() {
        assert_eq!(cos(29.0) * 17000.0, 14868.535021369727);
        assert_eq!(cos(29) * 17000.0, 14868.535021369727);
        assert_eq!(cos(30.0), 0.8660254037844387);
        assert_eq!(cos(45.0), std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(cos(60.0), 0.5000000000000001);
        assert_eq!(cos(60), 0.5000000000000001);
        assert_eq!(cos(60), cos(-60));
    }
    #[test]
    fn test_inverse_tan() {
        assert_eq!(inverse_tan(1.0), 45.0);
        assert_eq!(inverse_tan(1.0 / -2.0), -26.56505117707799);
        assert_eq!(inverse_tan(2.80 / 1.20), 66.80140948635182);
        assert_eq!(inverse_tan(87.6 / 309.7), 15.793787773268155);
        assert_eq!(inverse_tan(f64::sqrt(3.0)), 59.99999999999999);
        assert_eq!(inverse_tan(100.0 / -200.0), -26.56505117707799);
        assert_eq!(inverse_tan(1.0 / f64::sqrt(3.0)), 30.000000000000004);
    }
    #[test]
    fn test_display() {
        let expected_3d_vector_output = [
            (Vector([0.0, 2.200, 15.22]), "2.2j + 15.22k"),
            (Vector([-1.0, -1.0, -1.0]), "- i - j - k"),
            (Vector([-1.0, 1.0, -1.0]), "- i + j - k"),
            (Vector([3.0, -1.0, 4.0]), "3i - j + 4k"),
            (Vector([7.0, 1.0, -5.0]), "7i + j - 5k"),
            (Vector([1.0, -1.0, 1.0]), "i - j + k"),
            (Vector([1.0, 1.0, 1.0]), "i + j + k"),
            (Vector([-1.0, -1.0, 0.0]), "- i - j"),
            (Vector([-1.0, 1.0, 0.0]), "- i + j"),
            (Vector([1.0, -1.0, 0.0]), "i - j"),
            (Vector([1.0, 1.0, 0.0]), "i + j"),
            (Vector([1.0, 0.0, 1.0]), "i + k"),
            (Vector([-1.0, 0.0, 0.0]), "- i"),
            (Vector([0.0, -1.0, 0.0]), "- j"),
            (Vector([0.0, 0.0, -1.0]), "- k"),
            (Vector([0.0, 0.0, 0.0]), "0i"),
            (Vector([1.0, 0.0, 0.0]), "i"),
            (Vector([0.0, 1.0, 0.0]), "j"),
            (Vector([0.0, 0.0, 1.0]), "k"),
        ];
        let expected_2d_vector_output = [
            (Vector([-1.0, -1.0]), "- i - j"),
            (Vector([-1.0, 1.0]), "- i + j"),
            (Vector([1.0, -1.0]), "i - j"),
            (Vector([1.0, 1.0]), "i + j"),
            (Vector([-1.0, 0.0]), "- i"),
            (Vector([0.0, -1.0]), "- j"),
            (Vector([0.0, 0.0]), "0i"),
            (Vector([0.0, 1.0]), "j"),
            (Vector([1.0, 0.0]), "i"),
        ];
        verify(&expected_3d_vector_output);
        verify(&expected_2d_vector_output);
        assert_eq!(format!("{}", Vector([1., 2., 3., 4.])), "i + 2j + 3k + 4l");
        assert_eq!(
            format!("{}", Vector([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.])),
            "i + 2j + 3k + 4l + 5m + 6n + 7o + 8p + 9q + 10r + 11s + 12t + 13u + 14v + 15w + 16x + 17y + 18z"
        );
        assert_eq!(
            format!("{}", Vector([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])),
            "i + 2j + 3k + 4l + 5m + 6n + 7o + 8p + 9q + 10r + 11s + 12t + 13u + 14v + 15w + 16x + 17y + 18z + 19{ + 20|"
        );
    }
    #[test]
    fn test_dot() {
        assert_eq!(V1.dot(&V2), -162.4);
    }
    #[test]
    fn test_quadratic_equation() {
        assert_eq!(quadratic_equation(9.0, -126.0, 441.0), (7.0, 7.0));
        assert_eq!(quadratic_equation(1, 1, -6), (2.0, -3.0));
        assert_eq!(quadratic_equation(1, 1, 0), (0.0, -1.0));
        assert_eq!(quadratic_equation(1, 0, 0), (0.0, 0.0));
        assert_eq!(
            quadratic_equation(3.1725, -39.4875, 121.214),
            (6.946617853705707, 5.50019065693259),
        );
    }
    #[test]
    fn test_sin() {
        assert_eq!(sin(29.0) * 17000.0, 8241.76354418773);
        assert_eq!(sin(29) * 17000.0, 8241.76354418773);
        assert_eq!(sin(30.0), 0.49999999999999994);
        assert_eq!(sin(45.0), 0.7071067811865475);
        assert_eq!(sin(45), 0.7071067811865475);
        assert_eq!(sin(60.0), f64::sqrt(3.0) / 2.0);
    }
    #[test]
    fn test_sqrt() {
        assert_eq!(sqrt(0), 0.0);
        assert_eq!(sqrt(1), 1.0);
        assert_eq!(sqrt(1.0), 1.0);
        assert_eq!(sqrt(2), std::f64::consts::SQRT_2);
        assert_eq!(sqrt(2.0), std::f64::consts::SQRT_2);
        assert_eq!(sqrt(3.0), f64::sqrt(3.0));
        assert_eq!(sqrt(3), f64::sqrt(3.0));
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(sqrt(4), 2.0);
        assert_eq!(sqrt(9), 3.0);
    }
    #[test]
    fn test_tan() {
        assert_eq!(tan(29), 0.554309051452769);
        assert_eq!(tan(29.0), 0.554309051452769);
        assert_eq!(tan(30.0), 0.5773502691896257);
        assert_eq!(tan(45.0), 0.9999999999999999);
        assert_eq!(tan(60.0), 1.7320508075688767);
    }
    #[test]
    fn test_the_magnitude_of() {
        assert_eq!(the_magnitude_of(&Vector([0.0])), 0.0);
        assert_eq!(the_magnitude_of(&Vector([1.0])), 1.0);
        assert_eq!(the_magnitude_of(&Vector([-1.0])), 1.0);
        assert_eq!(the_magnitude_of(&Vector([3.0, 4.0])), 5.0);
        assert_eq!(the_magnitude_of(&Vector([6.0, 8.0])), 10.0);
        assert_eq!(the_magnitude_of(&Vector([3.0, 4.0, 0.0])), 5.0);
        assert_eq!(the_magnitude_of(&Vector([6.0, 8.0, 0.0])), 10.0);
        assert_eq!(the_magnitude_of(&Vector([3.0, 4.0, 0.0, 0.0])), 5.0);
        assert_eq!(the_magnitude_of(&Vector([6.0, 8.0, 0.0, 0.0])), 10.0);
        assert_eq!(the_magnitude_of(&Vector([0.0, 0.0, 3.0, 0.0, 4.0])), 5.0);
        assert_eq!(the_magnitude_of(&Vector([-7.0, 9.0])), 11.40175425099138);
        assert_eq!(the_magnitude_of(&Vector([-3.0, 5.0])), 5.830951894845301);
        assert_eq!(the_magnitude_of(&Vector([10.0, 23.0])), 25.079872407968907);
        assert_eq!(
            the_magnitude_of(&Vector([1.2000000000000002, 2.8000000000000003])),
            3.0463092423455636
        );

        assert_eq!(the_magnitude_of(Vector([0.0])), 0.0);
        assert_eq!(the_magnitude_of(Vector([1.0])), 1.0);
        assert_eq!(the_magnitude_of(Vector([-1.0])), 1.0);
        assert_eq!(the_magnitude_of(Vector([3.0, 4.0])), 5.0);
        assert_eq!(the_magnitude_of(Vector([6.0, 8.0])), 10.0);
        assert_eq!(the_magnitude_of(Vector([3.0, 4.0, 0.0])), 5.0);
        assert_eq!(the_magnitude_of(Vector([6.0, 8.0, 0.0])), 10.0);
        assert_eq!(the_magnitude_of(Vector([3.0, 4.0, 0.0, 0.0])), 5.0);
        assert_eq!(the_magnitude_of(Vector([6.0, 8.0, 0.0, 0.0])), 10.0);
        assert_eq!(the_magnitude_of(Vector([0.0, 0.0, 3.0, 0.0, 4.0])), 5.0);
        assert_eq!(the_magnitude_of(Vector([-7.0, 9.0])), 11.40175425099138);
        assert_eq!(the_magnitude_of(Vector([-3.0, 5.0])), 5.830951894845301);
        assert_eq!(the_magnitude_of(Vector([10.0, 23.0])), 25.079872407968907);
    }
    #[test]
    fn test_the_moment_of_inertia_of() {
        let r0 = [4.into(); 4];
        let m0 = [50.into(); 4];
        let m1 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02];
        let r1 = [0.25, 0.15, 0.05, 0.05, 0.15, 0.25];
        assert_eq!(the_moment_of_inertia_of(&m0, &r0), 3200.into());
        assert_eq!(the_moment_of_inertia_of(&m1, &r1), 0.0034999999999999996);
    }
    #[test]
    fn test_the_angle_between() {
        assert_eq!(the_angle_between(&V1, &V2), 116.02154864365895);
    }
    #[test]
    fn test_vector_addition() {
        let f1 = Vector([300.0, 0.0]);
        let f2 = Vector([0.0, 700.0]);
        let f3 = Vector([-500.0, 0.0]);
        let f4 = Vector([0.0, -600.0]);
        let v5 = Vector([-5.0, 0.0, 0.0]);
        let v6 = Vector([3.0, -1.0, 4.0]);
        let v7 = Vector([7.0, 1.0, -5.0]);
        assert_eq!(&v5 + Vector([0.0, 0.0, 0.0]), v5);
        assert_eq!(v6 + &v7, Vector([10.0, 0.0, -1.0]));
        assert_eq!(v7, Vector([7.0, 1.0, -5.0]));
        assert_eq!(f1 + f2 + f3 + f4, Vector([-200.0, 100.0]));
        assert_eq!(Vector([-5.5]) + Vector([5.5]), Vector([0.0]));
        assert_eq!(&Vector([-5.5]) + &Vector([5.5]), Vector([0.0]));
        assert_eq!(Vector([1.0, 2.0]) + Vector([3.0, 4.0]), Vector([4.0, 6.0]));
    }
    #[test]
    fn test_vector_division_by_a_scalar() {
        assert_eq!(Vector([-5.5]) / 5.5, Vector([-1.0]));
        assert_eq!(Vector([1.0, 2.0]) / 2.0, Vector([0.5, 1.0]));
        assert_eq!(Vector([3.0, 4.0]) / 2.0, Vector([1.5, 2.0]));
        assert_eq!(Vector([6.0, 8.0]) / 2.0, Vector([3.0, 4.0]));
        assert_eq!(Vector([1.234, 0.0]) / 2.0, Vector([0.617, 0.0]));
        assert_eq!(Vector([555.55]) / 5.0, Vector([111.10999999999999]));
        assert_eq!(Vector([3.0, 4.0, 0.0]) / 2.0, Vector([1.5, 2.0, 0.0]));
        assert_eq!(Vector([1.0, 2.0, 3.0]) / 2.0, Vector([0.5, 1.0, 1.5]));
    }
    #[test]
    fn test_vector_multiplication_by_a_scalar() {
        let m = 2100.0;
        let a = Vector([-9.8]);
        let f = m * a;
        assert_eq!(Vector([-20580.0]), f);
        assert_eq!(Vector([-5.5]) * 5.5, Vector([-30.25]));
        assert_eq!(Vector([1.0, 2.0]) * 2.0, Vector([2.0, 4.0]));
        assert_eq!(Vector([3.0, 4.0]) * 2.0, Vector([6.0, 8.0]));
        assert_eq!(Vector([6.0, 8.0]) * 2.0, Vector([12.0, 16.0]));
        assert_eq!(Vector([1.234, 0.0]) * 2.0, Vector([2.468, 0.0]));
        assert_eq!(Vector([3.0, 4.0, 0.0]) * 2.0, Vector([6.0, 8.0, 0.0]));
        assert_eq!(Vector([1.0, 2.0, 3.0]) * 2.0, Vector([2.0, 4.0, 6.0]));
    }
}
