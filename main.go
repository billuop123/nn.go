package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strconv"

	rl "github.com/gen2brain/raylib-go/raylib"
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
	// matZero(&newMat)
	matZero(&newMat)
	return newMat
}

func matZero(m *Mat) {
	matAlloc(m, m.rows, m.cols)
	for row := range m.rows {
		for col := range m.cols {
			m.data[row][col] = 0
		}
	}
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
	if m1.cols != m2.cols || m1.rows != m2.rows {
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
		nn.w[i] = matRandVal(nn.size[i+1], nn.size[i], -1, 1)
		nn.b[i] = matRandVal(nn.size[i+1], 1, -1, 1)
	}
}

func matRandVal(rows, cols int, low, high float64) Mat {
	m := Mat{rows: rows, cols: cols}
	matRand(&m, low, high)
	return m
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
		nn.z[i+1] = sum
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

func drawCost(cost []float32, maxEpochs int) {
	var maxCost float32 = 0
	for _, c := range cost {
		if maxCost <= c {
			maxCost = c
		}
	}
	var padding float32 = 40.0
	var h float32 = 370
	var w float32 = 720
	for i := 1; i < len(cost); i++ {
		x1 := padding + float32(i-1)/float32(maxEpochs)*w
		y1 := padding + (1-cost[i-1]/maxCost)*h
		x2 := padding + float32(i)/float32(maxEpochs)*w
		y2 := padding + (1-cost[i]/maxCost)*h
		rl.DrawLineEx(
			rl.Vector2{X: x1, Y: y1},
			rl.Vector2{X: x2, Y: y2},
			2,
			rl.Green,
		)
	}
}

func main() {
	var cost []float32
	w := 800
	h := 450
	rl.InitWindow(800, 450, "Custom Neuron Network for XOR")
	defer rl.CloseWindow()
	rl.SetTargetFPS(25)
	nn, err := nnForm([]int{2, 8, 1})
	if err != nil {
		fmt.Println(err)
	}
	inputs := []Mat{
		{rows: 2, cols: 1, data: [][]float64{{0}, {0}}},
		{rows: 2, cols: 1, data: [][]float64{{0}, {1}}},
		{rows: 2, cols: 1, data: [][]float64{{1}, {0}}},
		{rows: 2, cols: 1, data: [][]float64{{1}, {1}}},
	}
	targets := [][][]float64{
		{{0}},
		{{1}},
		{{1}},
		{{0}},
	}
	const epochs int = 10000
	L := len(nn.size) - 1
	lr := 0.1
	epoch := 0
	for !rl.WindowShouldClose() {
		for range 50 {
			if epoch > epochs {
				break
			}
			for i := range len(inputs) {
				if err = nnForward(nn, inputs[i]); err != nil {
					fmt.Println(err)
					return
				}
				val, dz := finalGrad(nn.a[L].data, targets[i], nn.a[L-1].data)
				nn.w[L-1] = matSub(nn.w[L-1], scaleMat(val, lr))
				for k := range nn.size[L] {
					a := nn.a[L].data[k][0]
					y := targets[i][k][0]
					nn.b[L-1].data[k][0] -= lr * (-2 * (y - a) * a * (1 - a))
				}
				dW, dB := hiddenGrad(dz, nn.w[:L-1], nn.a[:L-1], nn.z[1:], nn.w[L-1])
				for i := range len(dW) {
					nn.w[i] = matSub(nn.w[i], scaleMat(dW[i], lr))
					for k := range len(dB[i]) {
						nn.b[i].data[k][0] -= lr * dB[i][k]
					}
				}
			}
			var totalCost float64
			for i := range len(inputs) {
				if err = nnForward(nn, inputs[i]); err != nil {
					fmt.Println(err)
					return
				}
				c, _ := nnCost(nn.a[L], Mat{rows: 1, cols: 1, data: targets[i]})
				totalCost += c
			}
			if len(cost) > 10000 {
				cost = cost[1:]
			}
			cost = append(cost, float32(totalCost/4))
		}
		rl.BeginDrawing()
		e := strconv.Itoa(epoch)
		text := "Cost Visualizer epoch:" + e
		rl.DrawText(text, int32((w/2)-len(text)), int32(h/16), 20, rl.Red)
		rl.ClearBackground(rl.Black)
		drawCost(cost, epochs)
		rl.EndDrawing()
		epoch++
	}
	for i := range len(inputs) {
		nnOutput(nn, inputs[i])
	}
}

func nnOutput(nn *NN, input Mat) {
	fmt.Printf("\n%d %d->", int(input.data[0][0]), int(input.data[1][0]))
	if err := nnForward(nn, input); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Print(int(nn.a[len(nn.size)-1].data[0][0]))
}

func scaleMat(m Mat, val float64) Mat {
	newMat := matForm(m.rows, m.cols)
	for i := range m.rows {
		for j := 0; j < m.cols; j++ {
			newMat.data[i][j] = val * m.data[i][j]
		}
	}
	return newMat
}

func matSub(m1, m2 Mat) Mat {
	newMat := matForm(m1.rows, m1.cols)
	for i := 0; i < m1.rows; i++ {
		for j := 0; j < m1.cols; j++ {
			newMat.data[i][j] = m1.data[i][j] - m2.data[i][j]
		}
	}
	return newMat
}

func finalGrad(a, y, x [][]float64) (Mat, []float64) {
	nOut := len(a)
	nIn := len(x)
	newMat := matForm(nOut, nIn)
	dz := make([]float64, nOut)
	for i := range nOut {
		dz[i] = -2 * (y[i][0] - a[i][0]) * a[i][0] * (1 - a[i][0])
		for j := range nIn {
			newMat.data[i][j] = dz[i] * x[j][0]
		}
	}
	return newMat, dz
}

func hiddenGrad(delta []float64, w, a, z []Mat, wLast Mat) ([]Mat, [][]float64) {
	nIn := wLast.cols
	prevDelta := make([]float64, nIn)
	for j := range nIn {
		var acc float64
		for i := range wLast.rows {
			acc += wLast.data[i][j] * delta[i]
		}
		sig := matSig(z[0].data[j][0])
		prevDelta[j] = acc * sig * (1 - sig)
	}
	delta = prevDelta
	grads := make([]Mat, len(w))
	biasGrad := make([][]float64, len(w))
	for c := len(w) - 1; c >= 0; c-- {
		nOut := w[c].rows
		nIn := w[c].cols
		dW := matForm(nOut, nIn)
		dB := make([]float64, nOut)
		for i := range nOut {
			dB[i] = delta[i]
			for j := range nIn {
				dW.data[i][j] = delta[i] * a[c].data[j][0]
			}
		}
		grads[c] = dW
		biasGrad[c] = dB
		if c > 0 {
			prevDelta := make([]float64, nIn)
			for j := range nIn {
				var acc float64
				for i := range nOut {
					acc += w[c].data[i][j] * delta[i]
				}
				zVal := z[c].data[j][0]
				sig := matSig(zVal)
				prevDelta[j] = acc * sig * (1 - sig)
			}
			delta = prevDelta
		}
	}
	return grads, biasGrad
}
