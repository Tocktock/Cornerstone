import type { ActorSession, GraphEdgePayload, GraphSlicePayload, ResourceRef } from './types/api'

type GraphExplorerNode = ResourceRef & {
  isRoot: boolean
  inboundCount: number
  outboundCount: number
  totalRelationCount: number
}

type GraphExplorerRelation = GraphEdgePayload & {
  counterpart: ResourceRef
}

export type GraphExplorerSelection = {
  selectedNode: GraphExplorerNode
  nodes: GraphExplorerNode[]
  rootNodes: GraphExplorerNode[]
  inboundRelations: GraphExplorerRelation[]
  outboundRelations: GraphExplorerRelation[]
}

export function canActorReview(actor: ActorSession) {
  return actor.preferred_consumer_scope === 'review' || actor.base_role.toLowerCase() === 'admin'
}

export function canActorManageConnectors(actor: ActorSession) {
  return (
    actor.base_role.toLowerCase() === 'admin' ||
    actor.scoped_capabilities.some(
      (entry) => entry.capability === 'manage_connectors' && entry.scope === 'workspace',
    )
  )
}

export function buildGraphExplorer(
  payload: GraphSlicePayload,
  selectedNodeId?: string | null,
): GraphExplorerSelection | null {
  if (!payload.nodes.length) {
    return null
  }

  const rootIds = new Set(payload.root_concept_refs.map((root) => root.resource_id))
  const nodeOrder = payload.nodes
    .map((node) => {
      const inboundCount = payload.edges.filter(
        (edge) => edge.object_concept_ref.resource_id === node.resource_id,
      ).length
      const outboundCount = payload.edges.filter(
        (edge) => edge.subject_concept_ref.resource_id === node.resource_id,
      ).length
      return {
        ...node,
        isRoot: rootIds.has(node.resource_id),
        inboundCount,
        outboundCount,
        totalRelationCount: inboundCount + outboundCount,
      }
    })
    .sort((left, right) => {
      if (left.isRoot !== right.isRoot) {
        return left.isRoot ? -1 : 1
      }
      if (left.totalRelationCount !== right.totalRelationCount) {
        return right.totalRelationCount - left.totalRelationCount
      }
      return left.resource_label.localeCompare(right.resource_label)
    })

  const selectedNode =
    nodeOrder.find((node) => node.resource_id === selectedNodeId) ??
    nodeOrder.find((node) => node.isRoot) ??
    nodeOrder[0]

  const outboundRelations = payload.edges
    .filter((edge) => edge.subject_concept_ref.resource_id === selectedNode.resource_id)
    .map((edge) => ({
      ...edge,
      counterpart: edge.object_concept_ref,
    }))
    .sort((left, right) => left.counterpart.resource_label.localeCompare(right.counterpart.resource_label))

  const inboundRelations = payload.edges
    .filter((edge) => edge.object_concept_ref.resource_id === selectedNode.resource_id)
    .map((edge) => ({
      ...edge,
      counterpart: edge.subject_concept_ref,
    }))
    .sort((left, right) => left.counterpart.resource_label.localeCompare(right.counterpart.resource_label))

  return {
    selectedNode,
    nodes: nodeOrder,
    rootNodes: nodeOrder.filter((node) => node.isRoot),
    inboundRelations,
    outboundRelations,
  }
}

export function formatLocalDateTime(value?: string | null) {
  if (!value) {
    return 'Never'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed)
}
