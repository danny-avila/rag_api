import { expect, test } from "bun:test";
import { scopeWhere, liveWhere, vectorSQL } from "../src/clickhouse";

test("library/file predicates preserve correlated grants instead of cross-product access", () => {
  const where = scopeWhere([
    { tenantId: "tenant-a", namespaceId: "a", resourceIds: ["one"] },
    { tenantId: "tenant-a", namespaceId: "b", resourceIds: ["two"] },
  ]);
  expect(where.sql).toContain(
    "namespace_id = {lib0:String} AND file_id IN {files0:Array(String)}",
  );
  expect(where.sql).toContain(
    "namespace_id = {lib1:String} AND file_id IN {files1:Array(String)}",
  );
  expect(where.params).toEqual({
    tenant: "tenant-a",
    lib0: "a",
    files0: ["one"],
    lib1: "b",
    files1: ["two"],
  });
  expect(() =>
    scopeWhere([
      { tenantId: "a", namespaceId: "x" },
      { tenantId: "b", namespaceId: "x" },
    ]),
  ).toThrow();
});
test("publication is a scoped semi-join filter; the outer vector scan stays eligible", () => {
  const where = liveWhere(
    [{ tenantId: "a", namespaceId: "x", resourceIds: ["one"] }],
    "space",
  );
  const sql = vectorSQL(where.sql);
  expect(sql).toContain("FROM documents FINAL");
  expect(sql).toContain("state = 'ready'");
  expect(sql).toContain(
    "cosineDistance(embedding, {vector:Array(Float32)}) AS distance",
  );
  expect(sql).toContain("ORDER BY distance ASC LIMIT {k:UInt32}");
  expect(sql).not.toContain("FROM chunks FINAL");
  expect(sql).not.toContain(" JOIN ");
  expect(sql).not.toContain("LIMIT 1 BY");
});
