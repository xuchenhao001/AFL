package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
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
	IsSync bool `json:"is_sync"`
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	// generate a new uuid for each user
	var localMSPID string = os.Getenv("CORE_PEER_LOCALMSPID")
	println("LOCALMSPID: " + localMSPID)
	myuuid = strings.Trim(localMSPID, "OrgMSP")
	println("Init finished. My uuid: " + myuuid)
	return nil
}

// Start STEP #1
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

// UploadLocalModel STEP #2
func (s *SmartContract) UploadLocalModel(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[UPLOAD LOCAL MODEL MSG] Received")
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

// UploadGlobalModel STEP #3
func (s *SmartContract) UploadGlobalModel(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[UPLOAD GLOBAL MODEL MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	// store global model hash into the ledger
	err := saveAsMap(ctx, "globalModelHashMap", recMsg.Epochs, recMsg.Uuid, recMsg.Data)
	if err != nil {
		return fmt.Errorf("failed to save global model hash into state. %s", err.Error())
	}

	// if it is sync FL, trigger the next step
	// let each node download the latest global model
	if recMsg.IsSync == true {
		sendMsg := new(HttpMessage)
		sendMsg.Message = "global_model_update"
		sendMsg.Uuid = myuuid
		sendMsg.Epochs = recMsg.Epochs
		sendMsgAsBytes, _ := json.Marshal(sendMsg)
		go sendPostRequest(sendMsgAsBytes, "GLOBAL_MODEL_UPDATE")
	}
	return nil
}

// PrepareNextRound Prepare Next Round
func (s *SmartContract) PrepareNextRound(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[PREPARE NEXT ROUND MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	recMsg.Uuid = myuuid
	recMsg.Message = "prepare"
	sendMsgAsBytes, _ := json.Marshal(recMsg)

	go sendPostRequest(sendMsgAsBytes, "PREPARE")

	return nil
}

// ShutdownPython Shutdown the python process
func (s *SmartContract) ShutdownPython(ctx contractapi.TransactionContextInterface, receiveMsg string) error {
	fmt.Println("[SHUTDOWN PYTHON MSG] Received")
	receiveMsgBytes := []byte(receiveMsg)
	recMsg := new(HttpMessage)
	_ = json.Unmarshal(receiveMsgBytes, recMsg)

	recMsg.Uuid = myuuid
	recMsg.Message = "shutdown"
	sendMsgAsBytes, _ := json.Marshal(recMsg)

	go sendPostRequest(sendMsgAsBytes, "SHUTDOWN")

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
