package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
)

const url = "http://172.17.0.1:8888/messages"
var myuuid string
var userNum int

type SmartContract struct {
	contractapi.Contract
}

type HttpMessage struct {
	Message string `json:"message"`
	Data interface{} `json:"data"`
	Uuid string `json:"uuid"`
	Epochs int `json:"epochs"`
}

type HttpAccAlphaMessage struct {
	Message string `json:"message"`
	Data AccAlpha `json:"data"`
	Uuid string `json:"uuid"`
	Epochs int `json:"epochs"`
}

type AccAlpha struct {
	AccTest []float64 `json:"acc_test"`
	Alpha []float64 `json:"alpha"`
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	// generate a new uuid for each user
	var localMSPID string = os.Getenv("CORE_PEER_LOCALMSPID")
	println("LOCALMSPID: " + localMSPID)
	myuuid = strings.Trim(localMSPID, "OrgMSP")
	println("Init finished. My uuid: " + myuuid)
	return nil
}

// STEP #1
// (prepare for the training) BC-node1-python initiate local (global) model, and then send the hash of global model
// to the ledger.
func (s *SmartContract) Start(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[START MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)

	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// unmarshal to read user number
	dataMap := make(map[string]interface{})
	dataJson, err := json.Marshal(recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to marshal recMsg.Data interface: %s", err.Error())
	}
	err = json.Unmarshal(dataJson, &dataMap)
	if err != nil {
		return fmt.Errorf("failed to unmarshal dataJson to dataMap: %s", err.Error())
	}
	userNum = int(dataMap["user_number"].(float64))
	fmt.Println("Successfully loaded user number: ", userNum)
	// store initial global model hash into the ledger
	err = saveAsMap(ctx, "modelMap", recMsg.Epochs, "", recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to save model hash into state. %s", err.Error())
	}

	recMsg.Uuid = myuuid
	recMsg.Message = "prepare"
	sendMsgAsBytes, _ := json.Marshal(recMsg)

	go sendPostRequest(sendMsgAsBytes, "PREPARE")

	return nil
}

// STEP #2
// BC-nodes-python choose committee members according to global model hash, pull up hraftd distributed processes,
// send setup request to raftd and start up raft consensus，finally send raft network info to the ledger.
func (s *SmartContract) RaftInfo(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[RAFT INFO MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// store raft leader info into blockchain
	err := saveAsMap(ctx, "raftLeaderMap", recMsg.Epochs, "", recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to save raft leader info into state. %s", err.Error())
	}
	return nil
}

// STEP #3
// BC-nodes-python train local model based on previous round's local model, send local model to the committee leader,
// send hash of local model to the ledger.
func (s *SmartContract) Train(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[TRAIN MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// store local trained model hash into the ledger
	err := saveAsMap(ctx, "localModelHashMap", recMsg.Epochs, recMsg.Uuid, recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to save local model hash into state. %s", err.Error())
	}

	return nil
}

// STEP #4
// committee leader received all local models, aggregate to global model, then send the download link of global model
// and the hash of global model to the ledger.
func (s *SmartContract) GlobalModelUpdate(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[GLOBAL MODEL UPDATE MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// store global model hash into the ledger
	err := saveAsMap(ctx, "globalModelHashMap", recMsg.Epochs, recMsg.Uuid, recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to save global model hash into state. %s", err.Error())
	}

	// trigger STEP #5
	// let each node download the newest global model
	sendMsg := new(HttpMessage)
	sendMsg.Message = "global_model_update"
	sendMsg.Uuid = myuuid
	sendMsg.Epochs = recMsg.Epochs
	sendMsgAsBytes, _ := json.Marshal(sendMsg)
	go sendPostRequest(sendMsgAsBytes, "GLOBAL_MODEL_UPDATE")
	return nil
}

// Gather accuracy and alpha map from python:
// UserA {acc_test: [acc_test1, acc_test2, ...]
//        alpha: [alpha1, alpha2, ...]}
// UserB {acc_test: [acc_test1, acc_test2, ...]
//        alpha: [alpha1, alpha2, ...]}
func (s *SmartContract) AccAlphaMap(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[AccAlphaMap MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpAccAlphaMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// store acc_test and alpha map into blockchain
	err := saveAsMap(ctx, "accAlphaMap", recMsg.Epochs, recMsg.Uuid, recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to update acc_test and alpha map into state. %s", err.Error())
	}

	return nil
}

func (s *SmartContract) CheckAccAlphaMapRead(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[CHECK ACC ALPHA MAP READ MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpAccAlphaMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// try to read accAlphaMap, if all good, then can go on "negotiate ready".
	var accAlphaMap = map[string]AccAlpha{}
	accAlphaInterface, err := readAsMap(ctx, "accAlphaMap", recMsg.Epochs)
	if err != nil {
		return fmt.Errorf("failed to read acc_test and alpha map from state. %s", err.Error())
	}
	accAlphaString, err := json.Marshal(accAlphaInterface)
	if err != nil {
		return fmt.Errorf("failed to marshal accAlpha interface: %s", err.Error())
	}
	err = json.Unmarshal(accAlphaString, &accAlphaMap)
	if err != nil {
		return fmt.Errorf("failed to unmarshal accAlpha interface to accAlphaMap: %s", err.Error())
	}
	if len(accAlphaMap) == userNum {
		fmt.Println("gathered enough accuracy and alpha map [" + strconv.Itoa(len(accAlphaMap)) +
			"], can go on to negotiate ready now.")
	} else {
		fmt.Println("not gathered enough accuracy and alpha map [" + strconv.Itoa(len(accAlphaMap)) +
			"], do nothing.")
	}
	return nil
}

// STEP #6
// Smart Contract pick up the appropriate alpha according to the rule after gathering all alpha-accuracy maps, save
// to the ledger.
func (s *SmartContract) FindBestAlpha(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[FIND BEST ALPHA MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpAccAlphaMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	var accAlphaMap = map[string]AccAlpha{}
	accAlphaInterface, err := readAsMap(ctx, "accAlphaMap", recMsg.Epochs)
	if err != nil {
		return fmt.Errorf("failed to read acc_test and alpha map from state. %s", err.Error())
	}
	accAlphaString, err := json.Marshal(accAlphaInterface)
	if err != nil {
		return fmt.Errorf("failed to marshal accAlpha interface: %s", err.Error())
	}
	err = json.Unmarshal(accAlphaString, &accAlphaMap)
	if err != nil {
		return fmt.Errorf("failed to unmarshal accAlpha interface to accAlphaMap: %s", err.Error())
	}
	// count accAlpha map length. If gathered all of the acc_test, choose the best alpha according to the policy
	// (findMaxAccAvg or findMinAccVar), release alpha and w
	if len(accAlphaMap) == userNum {
		fmt.Println("gathered enough acc_test and alpha, choose the best alpha according to the policy")
		alpha, acc := findMaxAccAvg(accAlphaMap)
		// release alpha and accuracy
		data := make(map[string]interface{})
		data["alpha"] = alpha // alpha is included in data
		data["accuracy"] = acc // accuracy for alpha is included in data
		sendMsg := new(HttpMessage)
		sendMsg.Message = "best_alpha"
		sendMsg.Data = data
		sendMsg.Uuid = myuuid
		sendMsg.Epochs = recMsg.Epochs
		sendMsgAsBytes, _ := json.Marshal(sendMsg)

		go sendPostRequest(sendMsgAsBytes, "BEST_ALPHA")
	} else {
		fmt.Println("not gathered enough acc_test and alpha [" + strconv.Itoa(len(accAlphaMap)) + "], do nothing")
	}

	return nil
}

// sub-functions for STEP#6: find out the max acc_test average
func findMaxAccAvg(accAlphaMap map[string]AccAlpha) (float64, float64) {
	fmt.Println("[Find Alpha] According to max acc_test average policy")
	var accTestSum []float64
	var randomUuid string
	// calculate sum acc_test for all users into array `accTestSum`
	for id, accAlpha := range accAlphaMap {
		for k, v := range accAlpha.AccTest {
			if len(accTestSum) == k {
				accTestSum = append(accTestSum, 0)
			}
			accTestSum[k] += v
		}
		randomUuid = id
	}
	// find out the max value in `accTestSum`, return the alpha of that value.
	var max float64
	var maxIndex = 0
	for i, v := range accTestSum {
		if i==0 || v > max {
			max = v
			maxIndex = i
		}
	}
	alpha := accAlphaMap[randomUuid].Alpha[maxIndex]
	acc := max/float64(userNum)
	fmt.Println("Found the max acc_test: ", acc, " with alpha: ", alpha)
	return alpha, acc
}

// sub-functions for STEP#6: find out the min acc_test variance
func findMinAccVar(accAlphaMap map[string]AccAlpha) float64 {
	fmt.Println("[Find Alpha] According to min acc_test variance policy")
	var accTestAvg []float64
	var randomUuid string
	// calculate sum acc_test for all users into array `accTestSum`
	for id, accAlpha := range accAlphaMap {
		for k, accTest := range accAlpha.AccTest {
			if len(accTestAvg) == k {
				accTestAvg = append(accTestAvg, 0)
			}
			accTestAvg[k] += accTest / float64(userNum)
		}
		randomUuid = id
	}
	accTestVar :=make([]float64, len(accTestAvg))
	// calculate the variance value of acc_test
	for k, accAvg := range accTestAvg {
		nVariance := 0.0
		for _, accAlpha := range accAlphaMap {
			nVariance += math.Pow(accAlpha.AccTest[k] - accAvg, 2)
		}
		accTestVar[k] = nVariance / float64(userNum)
	}
	// find out the min variance value in accTestVar
	var min float64
	var minIndex = 0
	for i, v := range accTestVar {
		if i==0 || v < min {
			min = v
			minIndex = i
		}
	}
	alpha := accAlphaMap[randomUuid].Alpha[minIndex]
	fmt.Println("Found the min acc_test variance: ", min, " with alpha: ", alpha)
	return alpha
}

// Start Next Round: prepare committee
func (s *SmartContract) PrepareNextRoundCommittee(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[PREPARE NEXT ROUND COMMITTEE MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	recMsg.Uuid = myuuid
	recMsg.Message = "prepare"
	sendMsgAsBytes, _ := json.Marshal(recMsg)

	go sendPostRequest(sendMsgAsBytes, "PREPARE")

	return nil
}

func saveAsMap(ctx contractapi.TransactionContextInterface, keyType string, epochs int, myUUID string,
	value interface{}) error {
	epochsString := strconv.Itoa(epochs)
	fmt.Println("save [" + keyType + "] map to DB in epoch [" + epochsString  + "] for uuid: [" + myUUID + "]")

	key, err := ctx.GetStub().CreateCompositeKey(keyType, []string{epochsString, myUUID})
	if err !=nil {
		return fmt.Errorf("failed to composite key: %s", err.Error())
	}

	jsonAsBytes, _ := json.Marshal(value)
	err = ctx.GetStub().PutState(key, jsonAsBytes)
	if err != nil {
		return fmt.Errorf("failed to save map into state: %s", err.Error())
	}
	return nil
}

func readAsMap(ctx contractapi.TransactionContextInterface,
	keyType string, epochs int) (map[string]interface{}, error) {

	epochsString := strconv.Itoa(epochs)
	fmt.Println("read [" + keyType + "] map from DB in epoch [" + epochsString  + "]")

	mapIter, err := ctx.GetStub().GetStateByPartialCompositeKey(keyType, []string{epochsString})
	if err != nil {
		return nil, fmt.Errorf("failed to read map from state by partial composite key: %s", err.Error())
	}
	defer mapIter.Close()

	resultMap := make(map[string]interface{})

	for mapIter.HasNext() {
		mapItem, err := mapIter.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to read next map item from state: %s", err.Error())
		}

		var compositeKeyAttri []string
		_, compositeKeyAttri, err = ctx.GetStub().SplitCompositeKey(mapItem.Key)
		if err != nil {
			return nil, fmt.Errorf("failed to split composite key: %s", err.Error())
		}
		var myUUID string
		myUUID = compositeKeyAttri[1]
		valueMap := make(map[string]interface{})
		_ = json.Unmarshal(mapItem.Value, &valueMap)
		resultMap[myUUID] = valueMap
	}

	return resultMap, nil
}

func sendPostRequest(buf []byte, requestType string) {
	fmt.Println("SEND REQUEST [" + requestType + "]")
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(buf))
	if err != nil {
		fmt.Printf("[Error] failed to send post request to server. %s\n", err.Error())
		return
	}
	defer resp.Body.Close()
	if resp != nil {
		fmt.Println("SEND REQUEST [" + requestType + "]: " + resp.Status)
	} else {
		fmt.Println("SEND REQUEST [" + requestType + "]: No reply")
	}

}

func main() {

	chaincode, err := contractapi.NewChaincode(new(SmartContract))

	if err != nil {
		fmt.Printf("Error create chaincode: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting chaincode: %s", err.Error())
	}
}
