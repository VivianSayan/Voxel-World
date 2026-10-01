//! Growable integers.
//!
//! Two oracles, because a new numeric type cannot be checked against itself:
//!
//! 1. **[`WideUint`] and [`WideInt`]**, for every value small enough to fit one.
//!    Those are already checked against `u128` across their whole overlapping
//!    range, so they are a genuinely independent reference rather than a
//!    restatement — and they use a completely different representation (a fixed
//!    array, two's complement) reached by different code.
//! 2. **Python's arbitrary integers**, for the values no fixed width can hold:
//!    factorials, thousand-bit powers, and the divisions and gcds of them.
//!
//! The second is what actually tests the thing this type exists for. The first is
//! what catches an ordinary carry or borrow bug.

use std::str::FromStr;
use voxel_world::math::traits::{EuclideanRing, One, Semiring, Zero};
use voxel_world::math::{BigInt, BigUint, WideInt, WideUint};

/// Wide enough to hold anything the differential tests build, so a fit is never
/// the thing being tested.
type Wide = WideUint<8>; // 512 bits

/// Values far beyond any fixed width, computed with Python's arbitrary integers.
const KNOWN: &[(&str, &str)] = &[
    ("100!", "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000"),
    ("200!", "788657867364790503552363213932185062295135977687173263294742533244359449963403342920304284011984623904177212138919638830257642790242637105061926624952829931113462857270763317237396988943922445621451664240254033291864131227428294853277524242407573903240321257405579568660226031904170324062351700858796178922222789623703897374720000000000000000000000000000000000000000000000000"),
    ("2^1000", "10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376"),
    ("2^1000 - 1", "10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069375"),
    ("3^500", "36360291795869936842385267079543319118023385026001623040346035832580600191583895484198508262979388783308179702534403855752855931517013066142992430916562025780021771247847643450125342836565813209972590371590152578728008385990139795377610001"),
    ("10^100", "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
    ("(2^521-1) squared", "47125446914534694131579097993419809976955095716785201420286055195012674566357244479460731079205201122720511132925006540350105785156086431086764996857554304847155991333706718342307167456986269662311038377104760933477381254100896222805785374204495333936040246318307567782851014765052850751581472024524956029996236801"),
];

/// Divisions of very large values, with exact quotient and remainder.
const BIG_DIVISIONS: &[(&str, &str, &str, &str)] = &[
    ("93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000", "1606938044258990275541962092341162602522202993782792835301377", "58077046453262489576172766509561862644074070833915725568956775306606909088634596728769577563356345", "1341983381356805572777851679393776150974560920604900279812935"),
    ("10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376", "515377520732011331036461129765621272702107522001", "20790751712732071385260775389205038176518395689256962640553572190852257598673381784322742265316923818079968268215969189602199756251999333965132909951960289976500893760055198463215219092698883914498589405550491920067455174599775586168862012029023648238127", "36190046634790305115672313423993393168128537249"),
    ("36360291795869936842385267079543319118023385026001623040346035832580600191583895484198508262979388783308179702534403855752855931517013066142992430916562025780021771247847643450125342836565813209972590371590152578728008385990139795377610001", "5260135901548373507240989882880128665550339802823173859498280903068732154297080822113666536277588451226982968856178217713019432250183803863127814770651880849955223671128444598191663757884322717271293251735781375", "6912424408115942787343680822", "4819315264701215360828385113144103172898650524314384579392011781474607611518744177116609307929326548605708209030873562812336157777187809692462900762511860378260500255213647427260909248140596643990341400605319751"),
    ("788657867364790503552363213932185062295135977687173263294742533244359449963403342920304284011984623904177212138919638830257642790242637105061926624952829931113462857270763317237396988943922445621451664240254033291864131227428294853277524242407573903240321257405579568660226031904170324062351700858796178922222789623703897374720000000000000000000000000000000000000000000000000", "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000", "8450550186924629495838157093855404565441366722012461965560414732385728621597296137876420884621412468214583850753160522570648517967532382305454834505647607520427964189812887040005874546880020480000000000000000000000000", "0"),
    ("10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "1798465042647412146620280340569649349251249", "5560297121638573447808725956596668949102138087896243240310", "710077252173979846648660765914723925352810"),
    ("47125446914534694131579097993419809976955095716785201420286055195012674566357244479460731079205201122720511132925006540350105785156086431086764996857554304847155991333706718342307167456986269662311038377104760933477381254100896222805785374204495333936040246318307567782851014765052850751581472024524956029996236801", "6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151", "6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151", "0"),
];

/// Greatest common divisors: gcd(2^a - 1, 2^b - 1) = 2^gcd(a,b) - 1.
const GCDS: &[(&str, &str, &str)] = &[
    ("1606938044258990275541962092341162602522202993782792835301375", "1267650600228229401496703205375", "1267650600228229401496703205375"),
    ("7547924849643082704483109161976537781833842440832880856752412600491248324784297704172253450355317535082936750061527689799541169259849585265122868502865392087298790653951", "293567822846729153486185074598667128421960318613539983838411371441526128139326055432962374798096087878991871", "2251799813685247"),
    ("10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069375", "4149515568880992958512407863691161151012446232242436899995657329690652811412908146399707048947103794288197886611300789182395151075411775307886874834113963687061181803401509523685375", "1606938044258990275541962092341162602522202993782792835301375"),
    ("158456325028528675187087900671", "618970019642690137449562111", "1"),
];

fn big(text: &str) -> BigUint {
    BigUint::from_str(text).expect("the table holds valid digits")
}

/// A spread of `u128` values on the awkward boundaries, plus a fixed pseudorandom
/// set so a failure reproduces.
fn probes() -> Vec<u128> {
    let mut values: Vec<u128> = vec![
        0,
        1,
        2,
        u32::MAX as u128,
        u64::MAX as u128 - 1,
        u64::MAX as u128,
        u64::MAX as u128 + 1,
        1 << 64,
        1 << 127,
        u128::MAX / 2,
        u128::MAX - 1,
        u128::MAX,
    ];

    let mut state: u128 = 0xF00D_BABE_1234_5678_9ABC_DEF0_1122_3344;

    for _ in 0..30 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.push(state);
        // Truncated variants, so the limb count varies.
        values.push(state >> 64);
        values.push(state & u64::MAX as u128);
    }

    values
}

// ===========================================================================
// Against the fixed-width types
// ===========================================================================

#[test]
fn addition_agrees_with_the_fixed_width_type() {
    for left in probes() {
        for right in probes() {
            let grown: BigUint = BigUint::from(left).sum(&BigUint::from(right));
            let fixed: Wide = Wide::from(left)
                .checked_add(&Wide::from(right))
                .expect("512 bits has room");

            assert_eq!(grown.to_wide::<8>(), Some(fixed), "{left} + {right}");
        }
    }
}

#[test]
fn subtraction_agrees_with_the_fixed_width_type() {
    for left in probes() {
        for right in probes() {
            let grown: Option<BigUint> = BigUint::from(left).checked_sub(&BigUint::from(right));
            let fixed: Option<Wide> = Wide::from(left).checked_sub(&Wide::from(right));

            match (grown, fixed) {
                (Some(grown), Some(fixed)) => {
                    assert_eq!(grown.to_wide::<8>(), Some(fixed), "{left} - {right}");
                }
                (None, None) => {}
                (grown, fixed) => panic!("{left} - {right}: {grown:?} vs {fixed:?}"),
            }
        }
    }
}

#[test]
fn multiplication_agrees_with_the_fixed_width_type() {
    for left in probes() {
        for right in probes() {
            let grown: BigUint = BigUint::from(left).product(&BigUint::from(right));
            // 128 × 128 bits always fits in 512.
            let fixed: Wide = Wide::from(left)
                .checked_mul(&Wide::from(right))
                .expect("512 bits has room");

            assert_eq!(grown.to_wide::<8>(), Some(fixed), "{left} * {right}");
        }
    }
}

#[test]
fn division_agrees_with_the_fixed_width_type() {
    for left in probes() {
        for right in probes() {
            let grown = BigUint::from(left).div_rem(&BigUint::from(right));
            // The bit-at-a-time reference, not Algorithm D, so the two do not share
            // the algorithm being tested.
            let fixed = Wide::from(left).div_rem_binary(&Wide::from(right));

            match (grown, fixed) {
                (Some((quotient, remainder)), Some((expected_q, expected_r))) => {
                    assert_eq!(
                        quotient.to_wide::<8>(),
                        Some(expected_q),
                        "quotient of {left} / {right}"
                    );
                    assert_eq!(
                        remainder.to_wide::<8>(),
                        Some(expected_r),
                        "remainder of {left} / {right}"
                    );
                }
                (None, None) => {}
                (grown, fixed) => panic!("{left} / {right}: {grown:?} vs {fixed:?}"),
            }
        }
    }
}

#[test]
fn shifts_agree_with_the_fixed_width_type() {
    for value in probes() {
        for places in [0u64, 1, 7, 63, 64, 65, 100, 128, 200, 300] {
            let up: BigUint = BigUint::from(value).shifted_up(places);
            let down: BigUint = BigUint::from(value).shifted_down(places);

            assert_eq!(
                up.to_wide::<8>(),
                Some(Wide::from(value).wrapping_shl(places as u32)),
                "{value} << {places}"
            );
            assert_eq!(
                down.to_wide::<8>(),
                Some(Wide::from(value).wrapping_shr(places as u32)),
                "{value} >> {places}"
            );
        }
    }
}

#[test]
fn ordering_agrees_with_u128() {
    let probes: Vec<u128> = probes();

    for left in &probes {
        for right in &probes {
            assert_eq!(
                BigUint::from(*left).cmp(&BigUint::from(*right)),
                left.cmp(right),
                "comparing {left} with {right}"
            );
        }
    }
}

#[test]
fn signed_arithmetic_agrees_with_the_fixed_width_type() {
    let mut values: Vec<i128> = vec![0, 1, -1, 2, -2, i64::MAX as i128, i64::MIN as i128, i128::MAX, i128::MIN + 1];
    let mut state: u128 = 0xBEEF_CAFE_0011_2233_4455_6677_8899_AABB;

    for _ in 0..25 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.push(state as i128);
        values.push((state >> 70) as i128);
    }

    for left in &values {
        for right in &values {
            let grown_sum: BigInt = BigInt::from(*left).sum(&BigInt::from(*right));
            let grown_difference: BigInt = BigInt::from(*left).difference(&BigInt::from(*right));
            let grown_product: BigInt = BigInt::from(*left).product(&BigInt::from(*right));

            // The fixed type is 512 bits wide, so none of these overflow it.
            let wide_left: WideInt<8> = WideInt::from(*left);
            let wide_right: WideInt<8> = WideInt::from(*right);

            assert_eq!(
                grown_sum.to_string(),
                wide_left.checked_add(&wide_right).expect("fits").to_string(),
                "{left} + {right}"
            );
            assert_eq!(
                grown_difference.to_string(),
                wide_left.checked_sub(&wide_right).expect("fits").to_string(),
                "{left} - {right}"
            );
            assert_eq!(
                grown_product.to_string(),
                wide_left.checked_mul(&wide_right).expect("fits").to_string(),
                "{left} * {right}"
            );

            // And division, where the sign convention is easy to get backwards.
            match BigInt::div_rem(&BigInt::from(*left), &BigInt::from(*right)) {
                Some((quotient, remainder)) => {
                    assert_eq!(
                        quotient.to_i128(),
                        Some(left / right),
                        "{left} / {right}"
                    );
                    assert_eq!(
                        remainder.to_i128(),
                        Some(left % right),
                        "{left} % {right}"
                    );
                }
                None => assert_eq!(*right, 0, "only a zero divisor should refuse"),
            }
        }
    }
}

/// The fixed-width type needs a `checked_neg` that fails for `MIN`; this one does
/// not, because there is no `MIN`.
#[test]
fn negation_is_total() {
    for value in [0i128, 1, -1, i128::MIN, i128::MAX] {
        let grown: BigInt = BigInt::from(value);
        let negated: BigInt = grown.negated();

        assert_eq!(negated.negated(), grown, "negating twice returns {value}");

        if value != 0 {
            assert_ne!(negated, grown);
            assert_eq!(negated.is_negative(), !grown.is_negative());
        }
    }

    // `WideInt::MIN` has no negation; the grown equivalent does.
    assert_eq!(WideInt::<2>::MIN.checked_neg(), None);

    let grown_min: BigInt = BigInt::from_str(&WideInt::<2>::MIN.to_string()).unwrap();
    assert_eq!(
        grown_min.negated().to_string(),
        "170141183460469231731687303715884105728"
    );
}

// ===========================================================================
// Beyond any fixed width
// ===========================================================================

#[test]
fn computes_values_no_fixed_width_can_hold() {
    let mut factorial: BigUint = BigUint::one();
    for n in 1..=100u64 {
        factorial = factorial.product(&BigUint::from(n));
    }
    assert_eq!(factorial.to_string(), KNOWN[0].1, "100!");

    for n in 101..=200u64 {
        factorial = factorial.product(&BigUint::from(n));
    }
    assert_eq!(factorial.to_string(), KNOWN[1].1, "200!");

    assert_eq!(BigUint::from(2u64).pow(1000).to_string(), KNOWN[2].1, "2^1000");
    assert_eq!(
        BigUint::from(2u64)
            .pow(1000)
            .checked_sub(&BigUint::one())
            .unwrap()
            .to_string(),
        KNOWN[3].1,
        "2^1000 - 1"
    );
    assert_eq!(BigUint::from(3u64).pow(500).to_string(), KNOWN[4].1, "3^500");
    assert_eq!(BigUint::from(10u64).pow(100).to_string(), KNOWN[5].1, "10^100");

    // A 521-bit Mersenne prime squared, which needs 1042 bits.
    let mersenne: BigUint = BigUint::from(2u64)
        .pow(521)
        .checked_sub(&BigUint::one())
        .unwrap();
    assert_eq!(
        mersenne.product(&mersenne).to_string(),
        KNOWN[6].1,
        "(2^521-1)^2"
    );

    // None of these fit in even the widest fixed type used here.
    assert_eq!(factorial.to_wide::<8>(), None, "200! is far past 512 bits");
}

#[test]
fn divides_values_no_fixed_width_can_hold() {
    for (numerator, divisor, quotient, remainder) in BIG_DIVISIONS {
        let (found_quotient, found_remainder) = big(numerator)
            .div_rem(&big(divisor))
            .expect("not by zero");

        assert_eq!(
            found_quotient.to_string(),
            *quotient,
            "quotient of {numerator} / {divisor}"
        );
        assert_eq!(
            found_remainder.to_string(),
            *remainder,
            "remainder of {numerator} / {divisor}"
        );

        // And the identity, which needs no reference: q × d + r = n, with r < d.
        assert!(found_remainder < big(divisor));
        assert_eq!(
            found_quotient
                .product(&big(divisor))
                .sum(&found_remainder)
                .to_string(),
            *numerator
        );
    }
}

#[test]
fn finds_greatest_common_divisors_of_huge_values() {
    for (left, right, expected) in GCDS {
        assert_eq!(big(left).gcd(&big(right)).to_string(), *expected);

        // And through the signed trait, which is where the algorithm is provided.
        let signed_left: BigInt = BigInt::from_str(left).unwrap();
        let signed_right: BigInt = BigInt::from_str(right).unwrap();

        assert_eq!(
            signed_left.gcd_normalised(&signed_right).to_string(),
            *expected
        );
    }
}

#[test]
fn grows_without_any_ceiling() {
    // Repeated squaring from 2^1000 to 2^64000, which is 1000 limbs.
    let mut value: BigUint = BigUint::from(2u64).pow(1000);

    for _ in 0..6 {
        value = value.product(&value);
    }

    assert_eq!(value.bit_length(), 64_001);
    assert_eq!(value.limbs().len(), 1001);
    assert_eq!(value, BigUint::from(2u64).pow(64_000));

    // And it divides all the way back down.
    let (quotient, remainder) = value
        .div_rem(&BigUint::from(2u64).pow(63_999))
        .expect("not by zero");

    assert_eq!(quotient, BigUint::from(2u64));
    assert!(remainder.is_zero());
}

// ===========================================================================
// Division's rare correction
// ===========================================================================

/// Algorithm D's add-back step, which fires about once in `2^63` divisions and so
/// is never reached by random input. These are the same constructed shapes the
/// fixed-width tests use: a divisor with a large *low* limb, which the two-limb
/// estimate never looks at.
#[test]
fn handles_the_add_back_correction() {
    let base: u64 = u64::MAX;
    let half: u64 = 1 << 63;

    // Divisors of the shape [MAX, 0, 2^63] — normalised, with the largest possible
    // low limb.
    let divisor: BigUint = BigUint::from_limbs(vec![base, 0, half]);

    for numerator in [
        BigUint::from_limbs(vec![0, base - 1, half, half - 1]),
        BigUint::from_limbs(vec![base, base - 2, half, half - 1]),
        BigUint::from_limbs(vec![half - 2, half, half, 1 << 62]),
        BigUint::from_limbs(vec![base, base, base, base]),
    ] {
        let (quotient, remainder) = numerator.div_rem(&divisor).expect("not by zero");

        assert!(remainder < divisor, "the remainder must be smaller");
        assert_eq!(
            quotient.product(&divisor).sum(&remainder),
            numerator,
            "q × d + r should rebuild the numerator"
        );

        // And the fixed-width Algorithm D, independently reached, agrees.
        let wide_numerator: WideUint<4> = numerator.to_wide::<4>().expect("four limbs");
        let wide_divisor: WideUint<4> = divisor.to_wide::<4>().expect("three limbs");
        let (expected_q, expected_r) = wide_numerator
            .div_rem_binary(&wide_divisor)
            .expect("not by zero");

        assert_eq!(quotient.to_wide::<4>(), Some(expected_q));
        assert_eq!(remainder.to_wide::<4>(), Some(expected_r));
    }
}

// ===========================================================================
// Invariants
// ===========================================================================

/// Trailing zero limbs would break the derived `PartialEq`: a one-limb value
/// holding zero is the same number as the empty one but a different struct.
#[test]
fn never_keeps_a_trailing_zero_limb() {
    let cases: Vec<BigUint> = vec![
        BigUint::from_limbs(vec![0, 0, 0]),
        BigUint::from_limbs(vec![5, 0, 0]),
        BigUint::from(7u64).checked_sub(&BigUint::from(7u64)).unwrap(),
        BigUint::from(u64::MAX).product(&BigUint::zero()),
        BigUint::from(1u64).shifted_down(500),
        BigUint::from_limbs(vec![1, 1]).checked_sub(&BigUint::from_limbs(vec![0, 1])).unwrap(),
        big(KNOWN[2].1).div_rem(&big(KNOWN[2].1)).unwrap().1,
    ];

    for value in cases {
        assert_ne!(value.limbs().last(), Some(&0), "{value} kept a zero limb");
    }

    // So zero has exactly one spelling.
    assert_eq!(BigUint::from_limbs(vec![0, 0]), BigUint::zero());
    assert!(BigUint::from_limbs(vec![0, 0]).limbs().is_empty());
}

#[test]
fn has_no_negative_zero() {
    let zero: BigInt = BigInt::from_parts(true, BigUint::zero());

    assert_eq!(zero, BigInt::zero());
    assert!(!zero.is_negative());
    assert_eq!(zero.to_string(), "0");

    // And a cancelling subtraction gives the one canonical zero.
    let cancelled: BigInt = BigInt::from(-5).sum(&BigInt::from(5));
    assert_eq!(cancelled, BigInt::zero());
    assert!(!cancelled.is_negative());
}

// ===========================================================================
// Text
// ===========================================================================

#[test]
fn round_trips_through_text() {
    for (name, text) in KNOWN {
        let value: BigUint = big(text);

        assert_eq!(value.to_string(), *text, "{name}");
    }

    // And signed, with the sign.
    for text in ["0", "1", "-1", "-123456789012345678901234567890", KNOWN[0].1] {
        let value: BigInt = BigInt::from_str(text).expect("valid");
        assert_eq!(value.to_string(), text);
    }

    let negated: String = format!("-{}", KNOWN[1].1);
    assert_eq!(BigInt::from_str(&negated).unwrap().to_string(), negated);
}

#[test]
fn rejects_text_that_is_not_a_number() {
    for bad in ["", "abc", "12a", "1.5", " 1", "1 ", "--1", "+"] {
        assert!(BigUint::from_str(bad).is_err(), "{bad:?} should be refused");
    }

    // A sign is accepted only on the signed type.
    assert!(BigUint::from_str("-1").is_err());
    assert!(BigInt::from_str("-1").is_ok());
    assert!(BigInt::from_str("+1").is_ok());
    assert_eq!(BigInt::from_str("+1").unwrap(), BigInt::one());
}

#[test]
fn prints_hexadecimal_without_leading_zeros() {
    assert_eq!(format!("{:x}", BigUint::zero()), "0");
    assert_eq!(format!("{:x}", BigUint::from(255u64)), "ff");
    assert_eq!(
        format!("{:x}", BigUint::from_limbs(vec![1, 0xab])),
        "ab0000000000000001"
    );
}

// ===========================================================================
// Traits
// ===========================================================================

#[test]
fn the_ring_laws_hold_for_every_input() {
    // The point of the type: a fixed-width ring is only a ring until something
    // overflows. Here the laws hold for values of any size.
    let a: BigInt = BigInt::from_str(KNOWN[0].1).unwrap();
    let b: BigInt = BigInt::from_str(KNOWN[4].1).unwrap().negated();
    let c: BigInt = BigInt::from(2).pow(300);

    assert_eq!(a.sum(&b), b.sum(&a), "addition commutes");
    assert_eq!(a.product(&b), b.product(&a), "multiplication commutes");
    assert_eq!(
        a.sum(&b).sum(&c),
        a.sum(&b.sum(&c)),
        "addition associates exactly, with no rounding anywhere"
    );
    assert_eq!(
        a.product(&b.sum(&c)),
        a.product(&b).sum(&a.product(&c)),
        "multiplication distributes exactly"
    );
    assert_eq!(a.sum(&a.negated()), BigInt::zero());
}

#[test]
fn power_comes_from_the_semiring_trait() {
    // `Semiring::power` takes the trait route through the operators, and must agree
    // with the inherent `pow`.
    let three: BigUint = BigUint::from(3u64);

    assert_eq!(Semiring::power(&three, 500), three.pow(500));
    assert_eq!(Semiring::power(&three, 0), BigUint::one());

    let negative: BigInt = BigInt::from(-2);
    assert_eq!(Semiring::power(&negative, 101), negative.pow(101));
    assert!(negative.pow(101).is_negative(), "an odd power stays negative");
    assert!(!negative.pow(100).is_negative(), "an even power does not");
}

#[test]
fn zero_and_one_behave() {
    assert!(Zero::is_zero(&BigUint::zero()));
    assert!(One::is_one(&BigUint::one()));
    assert!(!Zero::is_zero(&BigUint::one()));
    assert_eq!(<BigUint as Zero>::zero(), BigUint::zero());
    assert_eq!(<BigInt as One>::one(), BigInt::one());
    assert!(!One::is_one(&BigInt::from(-1)), "minus one is not one");
}

#[test]
fn converts_to_and_from_the_fixed_width_types() {
    let value: WideUint<4> = WideUint::from(u128::MAX);
    let grown: BigUint = BigUint::from_wide(&value);

    assert_eq!(grown.to_wide::<4>(), Some(value));
    assert_eq!(grown.to_u128(), Some(u128::MAX));

    // Narrowing refuses rather than truncating.
    let too_big: BigUint = grown.product(&grown);
    assert_eq!(too_big.to_wide::<2>(), None, "256 bits will not fit in 128");
    assert_eq!(too_big.to_u128(), None);
    assert!(too_big.to_wide::<4>().is_some(), "but it does fit in 256");
}
