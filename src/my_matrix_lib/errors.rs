#[derive(Debug)]
pub enum MatrixError {
    SizeNotMatch(usize, usize),
    WidhtNotMatch,
    HeigthNotMach,
    NotInversible,
    IndexOutOfRange,
    ConversionError,
    Other(String),
}


impl Into<String> for &MatrixError {
    fn into(self) -> String {
        match self {
            MatrixError::SizeNotMatch(size1, size2) => {
                format!("The sizes does not matches ({} != {})", size1, size2)
            }
            MatrixError::HeigthNotMach => "The heights does not matches".to_string(),
            MatrixError::WidhtNotMatch => "The widths does not matches".to_string(),
            MatrixError::NotInversible => "The matrix is not inversible".to_string(),
            MatrixError::IndexOutOfRange => "The index is out of range".to_string(),
            MatrixError::ConversionError=> "Cannot convert types between each others".to_string(),

            MatrixError::Other(s) => format!("Other error :{}", s),
        }
    }
}

impl Into<String> for MatrixError {
    fn into(self) -> String {
        (&self).to_string()
    }
}

impl core::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", <&Self as Into<String>>::into(self))
    }
}

impl core::error::Error for MatrixError {}
