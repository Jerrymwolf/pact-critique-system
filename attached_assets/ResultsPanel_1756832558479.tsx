import { useState } from "react";
import { BarChart3, Download, Share, ChevronDown, ChevronUp, FileText, FileDown, BookOpen, ExternalLink, CheckSquare, Square, AlertCircle, CheckCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import CircularProgress from "@/components/ui/circular-progress";
import FeedbackCard from "@/components/FeedbackCard";
import SummaryAnalysisCard from "@/components/SummaryAnalysisCard";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { pdfExportService } from "@/utils/pdfExport";

import type { AnalysisResult } from "@shared/schema";

interface ResultsPanelProps {
  documentId: string;
}

interface AnalysisResponse {
  analysis: {
    id: string;
    status: string;
    overallScore: number;
    completedAt: string;
    analysisMode?: string;
  };
  results: AnalysisResult;
}

export default function ResultsPanel({ documentId }: ResultsPanelProps) {
  const [selectedTab, setSelectedTab] = useState("summary");
  const [selectedDimension, setSelectedDimension] = useState("1.0.0");
  const [showAllElements, setShowAllElements] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [showDetailedFindings, setShowDetailedFindings] = useState(false);
  const { toast } = useToast();

  const { data: analysisData, isLoading, error } = useQuery<AnalysisResponse>({
    queryKey: ['/api/documents', documentId, 'analysis', 'results'],
    enabled: !!documentId,
  });

  // Get document info for PDF export
  const { data: documentData } = useQuery<{ filename: string }>({
    queryKey: ['/api/documents', documentId],
    enabled: !!documentId,
  });

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-6 bg-gray-200 dark:bg-muted rounded w-1/3"></div>
            <div className="h-32 bg-gray-200 dark:bg-muted rounded"></div>
            <div className="h-4 bg-gray-200 dark:bg-muted rounded w-2/3"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !analysisData) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center py-8">
            <p className="text-text-secondary">
              Analysis results are not yet available. Please wait for the analysis to complete.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { results } = analysisData;
  const overallScore = results.overallScore || 0;
  
  // Get dimensions data
  const dimensionEntries = Object.entries(results.dimensions || {});
  const selectedDimensionData = results.dimensions[selectedDimension];
  
  // Get elements for selected dimension
  const selectedElements = selectedDimensionData?.sections ? 
    selectedDimensionData.sections.flatMap(sectionId => {
      const section = results.sections?.[sectionId];
      return section?.elements?.map(elementId => results.elements?.[elementId]).filter(Boolean) || [];
    }).filter(Boolean) : [];
  
  // Also get all elements for this dimension directly if sections don't work
  const directElements = Object.values(results.elements || {}).filter(element => 
    element?.dimension === selectedDimension
  );

  const getScoreColor = (score: number) => {
    if (score >= 4) return "text-success-green";
    if (score >= 3) return "text-warning-orange";
    return "text-critical-red";
  };

  const getScoreBackground = (score: number) => {
    if (score >= 4.5) return "bg-green-600"; // Exemplary
    if (score >= 3.5) return "bg-green-500"; // Strong
    if (score >= 2.5) return "bg-yellow-500"; // Competent
    if (score >= 1.5) return "bg-orange-500"; // Developing
    return "bg-red-500"; // Inadequate
  };

  const getScoreLevel = (score: number) => {
    if (score >= 4.5) return "Exemplary";
    if (score >= 3.5) return "Strong";
    if (score >= 2.5) return "Competent";
    if (score >= 1.5) return "Developing";
    return "Inadequate";
  };

  const handlePDFExport = async (exportType: 'default' | 'comprehensive' | 'custom' | 'checklist', options?: { includeChecklist?: boolean }) => {
    if (!analysisData) return;
    
    setIsExporting(true);
    try {
      const exportData = {
        analysis: {
          ...analysisData.analysis,
          analysisMode: analysisData.analysis.analysisMode || 'Basic'
        },
        results: analysisData.results,
        documentName: documentData?.filename
      };

      switch (exportType) {
        case 'default':
          await pdfExportService.exportToPDF(exportData, { 
            type: 'default', 
            includeChecklist: options?.includeChecklist 
          });
          toast({
            title: "PDF Export Successful",
            description: "Executive summary with findings checklist has been downloaded.",
          });
          break;
        case 'comprehensive':
          await pdfExportService.exportToPDF(exportData, { 
            type: 'comprehensive', 
            includeDetails: true,
            includeChecklist: options?.includeChecklist 
          });
          toast({
            title: "PDF Export Successful", 
            description: "Comprehensive report with findings checklist has been downloaded.",
          });
          break;
        case 'custom':
          const selectedDimensions = [selectedDimension];
          await pdfExportService.exportToPDF(exportData, { 
            type: 'custom', 
            includeDetails: true, 
            selectedDimensions 
          });
          toast({
            title: "PDF Export Successful",
            description: `Custom report has been downloaded for selected dimension.`,
          });
          break;
        case 'checklist':
          await pdfExportService.exportToPDF(exportData, { type: 'checklist' });
          toast({
            title: "PDF Export Successful",
            description: "Findings checklist has been downloaded for printing and revision tracking.",
          });
          break;
      }
    } catch (error) {
      console.error('PDF export failed:', error);
      toast({
        title: "Export Failed",
        description: "There was an error generating the PDF report. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary flex items-center">
            <BarChart3 className="text-success-green mr-2 h-5 w-5" />
            Analysis Results
          </h2>
          <div className="flex space-x-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" disabled={isExporting}>
                  <FileDown className="mr-1 h-4 w-4" />
                  {isExporting ? 'Generating...' : 'Export PDF'}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => handlePDFExport('default', { includeChecklist: true })}>
                  <FileText className="mr-2 h-4 w-4" />
                  Executive Summary
                  <span className="ml-2 text-xs text-gray-500">(Summary + checklist)</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handlePDFExport('comprehensive', { includeChecklist: true })}>
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Comprehensive Report
                  <span className="ml-2 text-xs text-gray-500">(Full details + checklist)</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handlePDFExport('checklist')}>
                  <CheckSquare className="mr-2 h-4 w-4" />
                  Checklist Only
                  <span className="ml-2 text-xs text-gray-500">(Print & track revisions)</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button variant="default" size="sm" className="bg-academic-blue hover:bg-academic-blue-dark">
              <Share className="mr-1 h-4 w-4" />
              Share
            </Button>
          </div>
        </div>

        {/* Learning Resources Banner */}
        <div className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <BookOpen className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-blue-800 dark:text-blue-200 mb-1">
                Improve Your Academic Writing Skills
              </h3>
              <p className="text-xs text-blue-700 dark:text-blue-300 mb-2">
                Each element in your analysis links to detailed explanations, examples, and improvement strategies in our PACT Wiki.
              </p>
              <div className="flex items-center space-x-3">
                <Button 
                  size="sm" 
                  variant="outline" 
                  className="text-xs h-7 text-blue-600 border-blue-300 hover:bg-blue-600 hover:text-white"
                  onClick={() => window.open('/pact-wiki', '_blank')}
                >
                  <ExternalLink className="mr-1 h-3 w-3" />
                  Explore PACT Wiki
                </Button>
                <span className="text-xs text-blue-600 dark:text-blue-400">
                  Click "Learn More" buttons below for targeted help
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Overall Score */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="text-center">
            <CircularProgress 
              value={overallScore} 
              max={5} 
              size={128}
              className="mx-auto mb-4"
            />
            <h3 className="text-lg font-semibold text-text-primary">Overall Assessment</h3>
          </div>

          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-text-primary mb-4">Dimension Scores</h3>
            
            {dimensionEntries.map(([dimensionId, dimension]) => (
              <div key={dimensionId} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">{dimension.name}</span>
                  <span className="text-xs font-medium text-text-secondary">
                    {getScoreLevel(dimension.score || 0)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-muted rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full transition-all duration-500 ${getScoreBackground(dimension.score || 0)}`}
                    style={{ width: `${((dimension.score || 0) / 5) * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Document Context Information */}
        {results.documentContext && (
          <div className="mb-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-purple-800 dark:text-purple-200 flex items-center">
                  <FileText className="mr-2 h-4 w-4" />
                  Document Context
                </h4>
                <span className="text-sm font-medium text-purple-600 dark:text-purple-300">
                  {results.documentContext.typeDescription}
                </span>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                {results.documentContext.academicLevel && (
                  <div>
                    <p className="text-text-secondary">Academic Level</p>
                    <p className="font-medium text-text-primary capitalize">
                      {results.documentContext.academicLevel}
                    </p>
                  </div>
                )}
                {results.documentContext.field && (
                  <div>
                    <p className="text-text-secondary">Field</p>
                    <p className="font-medium text-text-primary">
                      {results.documentContext.field}
                    </p>
                  </div>
                )}
                {results.documentContext.chapterNumber && (
                  <div>
                    <p className="text-text-secondary">Chapter</p>
                    <p className="font-medium text-text-primary">
                      Chapter {results.documentContext.chapterNumber}
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-text-secondary">Word Count</p>
                  <p className="font-medium text-text-primary">
                    {results.documentContext.wordCount.toLocaleString()} words
                  </p>
                </div>
                <div>
                  <p className="text-text-secondary">Feedback Tone</p>
                  <p className="font-medium text-text-primary">
                    {results.documentContext.feedbackTone}
                  </p>
                </div>
              </div>
              
              {results.documentContext.headerNote && (
                <div className="mt-3 pt-3 border-t border-purple-200 dark:border-purple-700">
                  <p className="text-xs text-purple-700 dark:text-purple-300">
                    {results.documentContext.headerNote}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analysis Coverage Metadata */}
        {results.analysisMetadata && (
          <div className="mb-8">
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-blue-800 dark:text-blue-200 flex items-center">
                  <FileText className="mr-2 h-4 w-4" />
                  Document Analysis Coverage
                </h4>
                <div className="text-sm text-blue-600 dark:text-blue-300">
                  {results.analysisMetadata.coveragePercentage}% analyzed
                </div>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-text-secondary">Document Length</p>
                  <p className="font-medium text-text-primary">
                    {results.analysisMetadata.documentLength.toLocaleString()} chars
                  </p>
                </div>
                <div>
                  <p className="text-text-secondary">Words Analyzed</p>
                  <p className="font-medium text-text-primary">
                    {results.analysisMetadata.analyzedWordCount.toLocaleString()} / {results.analysisMetadata.wordCount.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-text-secondary">Analysis Method</p>
                  <p className="font-medium text-text-primary">
                    {results.analysisMetadata.chunkingApplied ? 'Balanced Sampling' : 'Full Document'}
                  </p>
                </div>
                <div>
                  <p className="text-text-secondary">Coverage</p>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 dark:bg-muted rounded-full h-2">
                      <div 
                        className="h-2 rounded-full bg-blue-500"
                        style={{ width: `${results.analysisMetadata.coveragePercentage}%` }}
                      ></div>
                    </div>
                    <span className="text-xs font-medium">
                      {results.analysisMetadata.coveragePercentage}%
                    </span>
                  </div>
                </div>
              </div>
              
              {results.analysisMetadata.chunkingApplied && (
                <div className="mt-3 text-xs text-blue-600 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30 rounded p-2">
                  <strong>Note:</strong> Large document detected. Analysis uses balanced sampling across beginning, key academic sections (methodology, results, discussion), and conclusions to ensure comprehensive PACT evaluation.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Key Findings */}
        <div className="mb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-critical-red rounded-full"></div>
                <h4 className="font-semibold text-critical-red">Critical Issues</h4>
              </div>
              <p className="text-2xl font-bold text-critical-red mb-1">
                {results.summary?.criticalIssues || 0}
              </p>
              <p className="text-xs text-text-secondary">Require immediate attention</p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-warning-orange rounded-full"></div>
                <h4 className="font-semibold text-warning-orange">Areas for Improvement</h4>
              </div>
              <p className="text-2xl font-bold text-warning-orange mb-1">
                {results.summary?.improvements || 0}
              </p>
              <p className="text-xs text-text-secondary">Should be addressed</p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-success-green rounded-full"></div>
                <h4 className="font-semibold text-success-green">Strengths</h4>
              </div>
              <p className="text-2xl font-bold text-success-green mb-1">
                {results.summary?.strengths || 0}
              </p>
              <p className="text-xs text-text-secondary">Well-executed elements</p>
            </div>
          </div>

          {/* Detailed Findings Dropdown */}
          <div className="border border-gray-200 dark:border-border rounded-lg">
            <Button
              variant="ghost"
              onClick={() => setShowDetailedFindings(!showDetailedFindings)}
              className="w-full justify-between p-4 h-auto font-medium text-left hover:bg-gray-50 dark:hover:bg-muted/50 rounded-lg"
            >
              <span className="flex items-center">
                <FileText className="mr-2 h-4 w-4" />
                View Detailed Findings Checklist
              </span>
              {showDetailedFindings ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
            
            {showDetailedFindings && (
              <div className="border-t border-gray-200 dark:border-border p-4 bg-gray-50 dark:bg-muted/20">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Critical Issues Checklist */}
                  <div>
                    <h4 className="font-semibold text-critical-red mb-3 flex items-center">
                      <AlertCircle className="mr-2 h-4 w-4" />
                      Critical Issues ({(() => {
                        const criticalElements = Object.values(results.elements || {}).filter(el => 
                          el.score !== undefined && el.score < 2
                        );
                        return criticalElements.length;
                      })()})
                    </h4>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {Object.values(results.elements || {})
                        .filter(el => el.score !== undefined && el.score < 2)
                        .map((element, index) => (
                          <div key={index} className="flex items-start space-x-2 p-2 bg-red-50 dark:bg-red-900/20 rounded border-l-2 border-critical-red">
                            <Square className="h-4 w-4 text-critical-red mt-0.5 flex-shrink-0" />
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-critical-red">{element.name}</p>
                              {element.improvements && element.improvements.length > 0 && (
                                <p className="text-xs text-text-secondary mt-1">{element.improvements[0]}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      {Object.values(results.elements || {}).filter(el => 
                        el.score !== undefined && el.score < 2
                      ).length === 0 && (
                        <p className="text-sm text-text-secondary italic">No critical issues identified.</p>
                      )}
                    </div>
                  </div>

                  {/* Areas for Improvement Checklist */}
                  <div>
                    <h4 className="font-semibold text-warning-orange mb-3 flex items-center">
                      <AlertCircle className="mr-2 h-4 w-4" />
                      Areas for Improvement ({(() => {
                        const improvementElements = Object.values(results.elements || {}).filter(el => 
                          el.score !== undefined && el.score >= 2 && el.score < 4
                        );
                        return improvementElements.length;
                      })()})
                    </h4>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {Object.values(results.elements || {})
                        .filter(el => el.score !== undefined && el.score >= 2 && el.score < 4)
                        .map((element, index) => (
                          <div key={index} className="flex items-start space-x-2 p-2 bg-orange-50 dark:bg-orange-900/20 rounded border-l-2 border-warning-orange">
                            <Square className="h-4 w-4 text-warning-orange mt-0.5 flex-shrink-0" />
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-warning-orange">{element.name}</p>
                              {element.improvements && element.improvements.length > 0 && (
                                <p className="text-xs text-text-secondary mt-1">{element.improvements[0]}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      {Object.values(results.elements || {}).filter(el => 
                        el.score !== undefined && el.score >= 2 && el.score < 4
                      ).length === 0 && (
                        <p className="text-sm text-text-secondary italic">No areas for improvement identified.</p>
                      )}
                    </div>
                  </div>

                  {/* Strengths Checklist */}
                  <div>
                    <h4 className="font-semibold text-success-green mb-3 flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4" />
                      Strengths ({(() => {
                        const strengthElements = Object.values(results.elements || {}).filter(el => 
                          el.score !== undefined && el.score >= 4
                        );
                        return strengthElements.length;
                      })()})
                    </h4>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {Object.values(results.elements || {})
                        .filter(el => el.score !== undefined && el.score >= 4)
                        .map((element, index) => (
                          <div key={index} className="flex items-start space-x-2 p-2 bg-green-50 dark:bg-green-900/20 rounded border-l-2 border-success-green">
                            <Square className="h-4 w-4 text-success-green mt-0.5 flex-shrink-0" />
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-success-green">{element.name}</p>
                              {element.strengths && element.strengths.length > 0 && (
                                <p className="text-xs text-text-secondary mt-1">{element.strengths[0]}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      {Object.values(results.elements || {}).filter(el => 
                        el.score !== undefined && el.score >= 4
                      ).length === 0 && (
                        <p className="text-sm text-text-secondary italic">No key strengths identified yet.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Main Content Tabs */}
        <div className="border-t border-gray-200 dark:border-border pt-6">
          <Tabs value={selectedTab} onValueChange={setSelectedTab}>
            <TabsList className={`grid w-full ${results.customAnalysis ? 'grid-cols-3' : 'grid-cols-2'} mb-6`}>
              <TabsTrigger value="summary" className="flex items-center">
                <FileText className="mr-2 h-4 w-4" />
                Summary Analysis
              </TabsTrigger>
              <TabsTrigger value="details" className="flex items-center">
                <BarChart3 className="mr-2 h-4 w-4" />
                PACT Details
              </TabsTrigger>
              {results.customAnalysis && (
                <TabsTrigger value="custom" className="flex items-center">
                  <FileText className="mr-2 h-4 w-4" />
                  Custom Analysis
                </TabsTrigger>
              )}
            </TabsList>

            <TabsContent value="summary">
              {results.summaryAnalysis ? (
                <SummaryAnalysisCard summaryAnalysis={results.summaryAnalysis} />
              ) : (
                <div className="text-center py-8 text-text-secondary">
                  <p>Summary analysis is being generated...</p>
                  <p className="text-sm mt-1">This will be available shortly after the detailed analysis completes.</p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="details">
              <Tabs value={selectedDimension} onValueChange={setSelectedDimension}>
                <TabsList className="grid w-full grid-cols-5 mb-6">
                  {dimensionEntries.map(([dimensionId, dimension]) => (
                    <TabsTrigger 
                      key={dimensionId} 
                      value={dimensionId}
                      className="text-xs px-2"
                    >
                      {dimension.name.split(' ')[0]}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {dimensionEntries.map(([dimensionId, dimension]) => (
                  <TabsContent key={dimensionId} value={dimensionId}>
                    <div className="space-y-4">
                      <div className="mb-4">
                        <h3 className="text-lg font-semibold text-text-primary mb-2">
                          {dimension.name}
                        </h3>
                        <p className="text-sm text-text-secondary mb-4">
                          {dimension.description}
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className={`px-3 py-1 rounded-full ${getScoreBackground(dimension.score || 0)} text-white text-sm font-medium`}>
                              {(dimension.score || 0).toFixed(1)}/5
                            </div>
                            <span className="text-sm text-text-secondary">
                              {getScoreLevel(dimension.score || 0)}
                            </span>
                          </div>
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="text-academic-blue border-academic-blue hover:bg-academic-blue hover:text-white transition-colors"
                            onClick={() => {
                              window.open(`/pact-wiki/dimensions/${dimensionId}`, '_blank');
                            }}
                          >
                            <BookOpen className="mr-2 h-4 w-4" />
                            Learn About This Dimension
                          </Button>
                        </div>
                      </div>

                      {/* Element Feedback Cards */}
                      <div className="space-y-4">
                        {(selectedElements.length > 0 ? selectedElements : directElements)
                          .slice(0, showAllElements ? undefined : 3)
                          .map((element) => (
                            <FeedbackCard 
                              key={element.id} 
                              element={element} 
                              documentId={documentId}
                              documentName={documentData?.filename}
                            />
                          ))}
                      </div>

                      {/* No elements found message */}
                      {selectedElements.length === 0 && directElements.length === 0 && (
                        <div className="text-center py-8 text-text-secondary">
                          <p>No detailed feedback available for this dimension yet.</p>
                          <p className="text-sm mt-1">Analysis may still be in progress.</p>
                        </div>
                      )}

                      {/* Load More Button */}
                      {(selectedElements.length > 0 ? selectedElements : directElements).length > 3 && (
                        <div className="text-center mt-6">
                          <Button
                            variant="outline"
                            onClick={() => setShowAllElements(!showAllElements)}
                          >
                            {showAllElements ? (
                              <>
                                <ChevronUp className="mr-1 h-4 w-4" />
                                Show Less
                              </>
                            ) : (
                              <>
                                <ChevronDown className="mr-1 h-4 w-4" />
                                Load More Elements ({(selectedElements.length > 0 ? selectedElements : directElements).length - 3} remaining)
                              </>
                            )}
                          </Button>
                        </div>
                      )}
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </TabsContent>

            {results.customAnalysis && (
              <TabsContent value="custom">
                <div className="space-y-6">
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-3">
                          Your Custom Question
                        </h3>
                        <p className="text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30 rounded p-3 text-sm">
                          {results.customAnalysis.question}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-muted border border-gray-200 dark:border-border rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-text-primary mb-4">Analysis Response</h4>
                    <div className="prose dark:prose-invert max-w-none">
                      <p className="text-text-primary leading-relaxed whitespace-pre-wrap">
                        {results.customAnalysis.response}
                      </p>
                    </div>
                  </div>

                  {results.customAnalysis.keyFindings && results.customAnalysis.keyFindings.length > 0 && (
                    <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
                      <h4 className="text-lg font-semibold text-green-800 dark:text-green-200 mb-4">Key Findings</h4>
                      <ul className="space-y-2">
                        {results.customAnalysis.keyFindings.map((finding, index) => (
                          <li key={index} className="flex items-start space-x-3">
                            <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                            <span className="text-green-700 dark:text-green-300 text-sm">
                              {finding}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {results.customAnalysis.recommendations && results.customAnalysis.recommendations.length > 0 && (
                    <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-6">
                      <h4 className="text-lg font-semibold text-orange-800 dark:text-orange-200 mb-4">Recommendations</h4>
                      <ul className="space-y-2">
                        {results.customAnalysis.recommendations.map((recommendation, index) => (
                          <li key={index} className="flex items-start space-x-3">
                            <div className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></div>
                            <span className="text-orange-700 dark:text-orange-300 text-sm">
                              {recommendation}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </TabsContent>
            )}
          </Tabs>
        </div>
      </CardContent>
    </Card>
  );
}
