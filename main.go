package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

type Mat struct {
	rows int
	cols int
	data [][]float64
}

func matForm(rows, cols int) Mat {
	var newMat Mat
	newMat.rows = rows
	newMat.cols = cols
	matRand(&newMat, -1, 1)
	return newMat
}

func matAt(m *Mat, row, col int) *float64 {
	if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		return nil
	}
	return &m.data[row][col]
}

func matSig(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func matPrint(m *Mat) {
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			fmt.Printf("%f ", *matAt(m, row, col))
		}
		fmt.Println()
	}
}

func matDot(m1 Mat, m2 Mat) (Mat, error) {
	var err error
	if m1.cols != m2.rows {
		err = errors.New("matDot:Dimensions doesnot match")
		return Mat{}, err
	}
	newMat := matForm(m1.rows, m2.cols)
	for i := 0; i < newMat.rows; i++ {
		for j := 0; j < newMat.cols; j++ {
			for k := 0; k < m1.cols; k++ {
				*matAt(&newMat, i, j) += *matAt(&m1, i, k) * (*matAt(&m2, k, j))
			}
		}
	}
	return newMat, nil
}

func matRand(m *Mat, low float64, high float64) {
	matAlloc(m, m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			random := low + rand.Float64()*(high-low)
			m.data[i][j] = random
		}
	}
}

func matSum(m1 Mat, m2 Mat) (Mat, error) {
	var err error
	if m1.cols != m2.cols && m1.rows != m2.rows {
		err = errors.New("matSum:dimensions donot match")
		return Mat{}, err
	}
	newMat := matForm(m1.rows, m2.cols)
	for row := 0; row < m1.rows; row++ {
		for col := 0; col < m1.cols; col++ {
			*matAt(&newMat, row, col) = *matAt(&m1, row, col) + (*matAt(&m2, row, col))
		}
	}
	return newMat, nil
}

func matAlloc(m *Mat, rows, cols int) {
	m.data = make([][]float64, rows)
	for r := range rows {
		m.data[r] = make([]float64, cols)
	}
}

func act(m Mat) Mat {
	newMat := matForm(m.rows, m.cols)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			*matAt(&newMat, row, col) = matSig(*matAt(&m, row, col))
		}
	}
	return newMat
}

type NN struct {
	size []int
	a    []Mat // count
	b    []Mat // count-1
	w    []Mat // count -1
	z    []Mat //count-1
}

func nnForm(size []int) (*NN, error) {
	n := len(size)
	if n < 2 {
		err := errors.New("nnForm:There should be atleast 2 layers")
		return nil, err
	}
	nn := NN{
		size: size,
		a:    make([]Mat, n),
		b:    make([]Mat, n-1),
		w:    make([]Mat, n-1),
		z:    make([]Mat, n),
	}
	nnRand(&nn)
	return &nn, nil
}

func nnRand(nn *NN) {
	for i := 0; i < len(nn.size); i++ {
		nn.a[i] = matForm(nn.size[i], 1)
		nn.z[i] = matForm(nn.size[i], 1)
	}
	for i := 0; i < len(nn.size)-1; i++ {
		nn.w[i] = matForm(nn.size[i+1], nn.size[i])
		nn.b[i] = matForm(nn.size[i+1], 1)
	}
}

func nnPrint(nn *NN) {
	for i := 0; i < len(nn.size); i++ {
		fmt.Printf("a%d:\n", i)
		matPrint(&nn.a[i])
		if i == len(nn.size)-1 {
			continue
		}
		fmt.Printf("w%d:\n", i)
		matPrint(&nn.w[i])
		fmt.Printf("b%d:\n", i)
		matPrint(&nn.b[i])
	}
}

func nnForward(nn *NN, input Mat) error {
	nn.a[0] = input
	for i := 0; i < len(nn.size)-1; i++ {
		dot, err := matDot(nn.w[i], nn.a[i])
		if err != nil {
			return err
		}
		sum, err := matSum(dot, nn.b[i])
		if err != nil {
			return err
		}
		nn.z[i] = sum
		nn.a[i+1] = act(sum)
	}
	return nil
}

func nnCost(ti Mat, to Mat) (float64, error) {
	if ti.cols != to.cols || ti.rows != to.rows {
		err := errors.New("nnCost:Dimensions are not correct")
		return 0.0, err
	}
	var mse float64
	total := ti.rows * ti.cols
	for i := 0; i < ti.rows; i++ {
		for j := 0; j < ti.cols; j++ {
			diff := ti.data[i][j] - to.data[i][j]
			mse += diff * diff
		}
	}
	return mse / float64(total), nil
}

func main() {
	nn, err := nnForm([]int{2, 2, 3, 1})
	if err != nil {
		fmt.Println(err)
	}
	nnPrint(nn)
	input := Mat{
		rows: 2,
		cols: 1,
		data: [][]float64{
			{0},
			{1},
		},
	}
	err = nnForward(nn, input)
	if err != nil {
		fmt.Println(err)
	}
	cost, err := nnCost(
		nn.a[len(nn.size)-1],
		Mat{
			rows: 1,
			cols: 1,
			data: [][]float64{
				{1},
			},
		},
	)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Print(cost)
	matrix := nn.a[len(nn.size)-1]
	val := gradientCalc(matrix.data, [][]float64{{1}}, nn.a[len(nn.size)-2].data)
	fmt.Print(val)
}

func gradientCalc(a, y, x [][]float64) [][]float64 {
	nOut := len(a)
	nIn := len(x)
	dw := make([][]float64, nOut)
	for i := range nOut {
		dw[i] = make([]float64, nIn)
		for j := range nIn {
			dw[i][j] = -2 * (y[i][0] - a[i][0]) * a[i][0] * (1 - a[i][0]) * x[i][0]
		}
	}
	return dw
}
