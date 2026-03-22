package main

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	flightsql "github.com/apache/arrow-go/v18/arrow/flight/flightsql"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func failf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func main() {
	if len(os.Args) != 3 {
		failf("usage: %s <host:port> <query>", os.Args[0])
	}

	client, err := flightsql.NewClient(
		os.Args[1],
		nil,
		nil,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		failf("connect flight sql client: %v", err)
	}
	defer client.Close()

	var flightInfo *flight.FlightInfo
	for attempt := 0; attempt < 20; attempt++ {
		result, execErr := client.Execute(context.Background(), os.Args[2])
		if execErr == nil {
			flightInfo = result
			break
		}
		if !strings.Contains(execErr.Error(), "code = Unavailable") {
			failf("execute query: %v", execErr)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if flightInfo == nil {
		failf("execute query: server remained unavailable")
	}
	if len(flightInfo.Endpoint) != 1 || flightInfo.Endpoint[0].Ticket == nil {
		failf("expected exactly one endpoint with a ticket, got %d", len(flightInfo.Endpoint))
	}

	reader, err := client.DoGet(context.Background(), flightInfo.Endpoint[0].Ticket)
	if err != nil {
		failf("do_get: %v", err)
	}
	defer reader.Release()

	expectedRows := []string{"2:", "3:three", "4:<null>"}
	var gotRows []string
	var batchCount int
	var schemaChecked bool

	for reader.Next() {
		record := reader.Record()
		batchCount++

		if !schemaChecked {
			schema := record.Schema()
			if len(schema.Fields()) != 2 {
				failf("expected 2 fields, got %d", len(schema.Fields()))
			}
			if schema.Field(0).Name != "a" || schema.Field(0).Type.ID() != arrow.INT64 || schema.Field(0).Nullable {
				failf("unexpected field 0: %+v", schema.Field(0))
			}
			if schema.Field(1).Name != "s" || schema.Field(1).Type.ID() != arrow.STRING || !schema.Field(1).Nullable {
				failf("unexpected field 1: %+v", schema.Field(1))
			}
			schemaChecked = true
		}

		aValues, ok := record.Column(0).(*array.Int64)
		if !ok {
			failf("column 0 is not Int64: %T", record.Column(0))
		}
		sValues, ok := record.Column(1).(*array.String)
		if !ok {
			failf("column 1 is not Utf8: %T", record.Column(1))
		}

		for row := 0; row < int(record.NumRows()); row++ {
			switch {
			case sValues.IsNull(row):
				gotRows = append(gotRows, fmt.Sprintf("%d:<null>", aValues.Value(row)))
			default:
				gotRows = append(gotRows, fmt.Sprintf("%d:%s", aValues.Value(row), sValues.Value(row)))
			}
		}
	}

	if err := reader.Err(); err != nil {
		failf("stream read error: %v", err)
	}
	if !schemaChecked {
		failf("query returned no schema-bearing batches")
	}
	if !reflect.DeepEqual(gotRows, expectedRows) {
		failf("unexpected rows: got=%v expected=%v", gotRows, expectedRows)
	}

	fmt.Printf("ok batches=%d rows=%d\n", batchCount, len(gotRows))
}
